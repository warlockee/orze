import errno
import json
import logging
import os
import shutil
import socket
import time
import uuid
from pathlib import Path
from typing import Optional

from orze.engine.process import TrainingProcess
from orze.core.fs import tail_file

logger = logging.getLogger("orze")

# Errno codes for filesystem health checks
_FS_RETRIABLE_ERRNOS = {errno.EROFS, errno.ENOSPC, errno.EIO, errno.ESTALE}


def check_stalled(tp: TrainingProcess, stall_minutes: int) -> bool:
    """Check if a process has stalled (no log growth). Returns True if stalled."""
    if stall_minutes <= 0:
        return False

    import time
    now = time.time()
    try:
        current_size = tp.log_path.stat().st_size
    except OSError:
        return False

    if current_size > tp._last_log_size:
        tp._last_log_size = current_size
        tp._last_log_check = now
        tp._stall_since = 0.0
        return False

    if tp._stall_since == 0.0:
        tp._stall_since = now
    elif (now - tp._stall_since) > stall_minutes * 60:
        return True

    return False


def detect_fatal_in_log(tp: TrainingProcess) -> Optional[str]:
    """Check log tail for fatal errors in a still-running process.

    Returns the matched error snippet if found, else None.
    The process may hang after printing a fatal error (e.g. OOM, NCCL
    timeout, segfault message) without exiting.  Rather than hardcoding
    a fixed pattern list, we look for any Python exception that was
    printed but the process is still alive -- a sign it's hung.
    """
    tail = tail_file(tp.log_path, 8192)
    if not tail:
        return None
    # Look for a Python traceback followed by no further progress
    # (the traceback itself is in the last 8KB of the log)
    tb_idx = tail.rfind("Traceback (most recent call last)")
    if tb_idx == -1:
        return None
    # Extract from the traceback to the end
    snippet = tail[tb_idx:]
    # Only flag if there's very little output after the traceback
    # (< 200 chars -- just the error message, no further training output)
    lines_after_tb = snippet.split("\n")
    if len(lines_after_tb) > 15:
        # Lots of output after the traceback -- process recovered
        return None
    return snippet[:500]


def check_disk_space(path: Path, min_gb: float) -> bool:
    """Return True if disk has at least min_gb free. False if low."""
    if min_gb <= 0:
        return True
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024 ** 3)
        return free_gb >= min_gb
    except Exception:
        return True


def _adaptive_stall_minutes(results_dir: Path, configured: int) -> int:
    """Compute adaptive stall timeout: min(configured, max(5, 2x median training time)).
    Falls back to configured value if not enough data."""
    if configured <= 0:
        return 0
    times = []
    try:
        for d in results_dir.iterdir():
            if not d.is_dir() or not d.name.startswith("idea-"):
                continue
            mp = d / "metrics.json"
            if not mp.exists():
                continue
            m = json.loads(mp.read_text(encoding="utf-8"))
            if m.get("status") == "COMPLETED" and m.get("training_time"):
                times.append(m["training_time"])
    except Exception:
        pass
    if len(times) < 3:
        return configured
    times.sort()
    median_min = times[len(times) // 2] / 60.0
    adaptive = max(configured // 2, int(median_min * 2 + 0.5))
    effective = min(configured, adaptive)
    if effective != configured:
        logger.debug("Adaptive stall: median=%.1fm -> effective=%dm (configured=%dm)",
                     median_min, effective, configured)
    return effective


def fs_check_writable(path: Path) -> bool:
    """Verify the filesystem at *path* is writable by writing a probe file.
    Returns True if writable, False on EROFS/ENOSPC/EIO/ESTALE."""
    probe = path / f".orze_probe_{socket.gethostname()}_{os.getpid()}.tmp"
    try:
        fd = os.open(str(probe), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
        try:
            os.write(fd, b"ok")
            os.fsync(fd)
        finally:
            os.close(fd)
        # Read it back to verify
        content = probe.read_bytes()
        probe.unlink(missing_ok=True)
        return content == b"ok"
    except OSError as e:
        if e.errno in _FS_RETRIABLE_ERRNOS:
            logger.error("Filesystem health check FAILED at %s: %s (errno %d)",
                         path, os.strerror(e.errno), e.errno)
        else:
            logger.warning("Filesystem probe error at %s: %s", path, e)
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def fs_startup_check(path: Path) -> bool:
    """Full startup filesystem validation: write, read-back, delete.
    Raises SystemExit if the shared filesystem is not usable."""
    if not path.exists():
        try:
            path.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.error("Cannot create results directory %s: %s", path, e)
            return False

    probe = path / f".orze_startup_probe_{uuid.uuid4().hex[:8]}.tmp"
    token = uuid.uuid4().hex
    try:
        probe.write_text(token, encoding="utf-8")
        readback = probe.read_text(encoding="utf-8")
        probe.unlink()
        if readback != token:
            logger.error("Filesystem integrity check FAILED: wrote %r, read %r",
                         token, readback)
            return False
        return True
    except OSError as e:
        logger.error("Filesystem startup check FAILED at %s: %s", path, e)
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            pass
        return False


def cleanup_stale_locks(results_dir: Path, hostname: str):
    """Remove stale lock directories that belong to our hostname.
    Called at startup to recover from unclean shutdowns."""
    cleaned = 0
    now = time.time()
    try:
        for entry in results_dir.iterdir():
            if not entry.is_dir() or not entry.name.startswith("_"):
                continue
            if not entry.name.endswith("_lock"):
                continue
            lock_meta = entry / "lock.json"
            if not lock_meta.exists():
                continue
            try:
                meta = json.loads(lock_meta.read_text(encoding="utf-8"))
                if meta.get("host") == hostname:
                    age = now - meta.get("time", 0)
                    logger.info("Cleaning stale lock from our host: %s (age %.0fs)",
                                entry.name, age)
                    shutil.rmtree(entry, ignore_errors=True)
                    cleaned += 1
            except (json.JSONDecodeError, OSError):
                continue
    except OSError:
        pass
    if cleaned:
        logger.info("Cleaned %d stale lock(s) from %s", cleaned, hostname)
    return cleaned


class HealthMonitor:
    """Lightweight per-iteration filesystem health monitor."""

    def __init__(self, results_dir: Path, retry_delay: float = 30.0):
        self.results_dir = results_dir
        self.retry_delay = retry_delay
        self._last_fail_time: float = 0.0
        self._consecutive_fails: int = 0

    def check_before_write(self) -> bool:
        """Check filesystem health before coordination writes.
        Returns True if OK to proceed, False if should pause."""
        ok = fs_check_writable(self.results_dir)
        if ok:
            if self._consecutive_fails > 0:
                logger.info("Filesystem recovered after %d failed check(s)",
                            self._consecutive_fails)
            self._consecutive_fails = 0
            return True

        self._consecutive_fails += 1
        self._last_fail_time = time.time()
        logger.error("Filesystem unhealthy (fail #%d). Pausing %.0fs before retry.",
                     self._consecutive_fails, self.retry_delay)
        return False

    @property
    def healthy(self) -> bool:
        return self._consecutive_fails == 0
