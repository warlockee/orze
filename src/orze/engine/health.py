import json
import logging
import shutil
from pathlib import Path
from typing import Optional

from orze.engine.process import TrainingProcess
from orze.core.fs import tail_file

logger = logging.getLogger("orze")


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
