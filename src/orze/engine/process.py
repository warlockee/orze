import os
import signal
import subprocess
import logging
from dataclasses import dataclass, field
from typing import Any
from pathlib import Path

logger = logging.getLogger("orze")


def _kill_pg(proc: subprocess.Popen, sig=signal.SIGTERM):
    """Send a signal to the entire process group of *proc*.
    lookup fails (e.g. process already dead).
    """
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass


def _terminate_and_reap(proc: subprocess.Popen, label: str = "",
                        timeout: float = 10):
    """SIGTERM the process group, wait, then SIGKILL if needed."""
    _kill_pg(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("Force killing %s (PID %d)", label or "process", proc.pid)
        _kill_pg(proc, signal.SIGKILL)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.error("Failed to reap %s (PID %d) after SIGKILL",
                         label or "process", proc.pid)


# ---------------------------------------------------------------------------
# Utilities

@dataclass
class TrainingProcess:
    """Tracks a non-blocking training subprocess."""
    idea_id: str
    gpu: int
    process: subprocess.Popen
    start_time: float
    log_path: Path
    timeout: float
    _log_fh: Any = field(default=None, repr=False)
    _last_log_size: int = field(default=0, repr=False)
    _last_log_check: float = field(default=0.0, repr=False)
    _stall_since: float = field(default=0.0, repr=False)

    def close_log(self):
        """Close the log file handle if open."""
        if self._log_fh and not self._log_fh.closed:
            try:
                self._log_fh.close()
            except Exception:
                pass


@dataclass
class EvalProcess:
    """Tracks a non-blocking eval subprocess."""
    idea_id: str
    gpu: int
    process: subprocess.Popen
    start_time: float
    log_path: Path
    timeout: float
    _log_fh: Any = field(default=None, repr=False)

    def close_log(self):
        if self._log_fh and not self._log_fh.closed:
            try:
                self._log_fh.close()
            except Exception:
                pass


@dataclass
class RoleProcess:
    """Tracks a non-blocking agent role subprocess."""
    role_name: str
    process: subprocess.Popen
    start_time: float
    log_path: Path
    timeout: float
    lock_dir: Path
    cycle_num: int
    _log_fh: Any = field(default=None, repr=False)
    ideas_pre_size: int = 0  # ideas.md size before role started
    ideas_pre_count: int = 0  # idea count before role started

    def close_log(self):
        if self._log_fh and not self._log_fh.closed:
            try:
                self._log_fh.close()
            except Exception:
                pass


def run_pre_script(idea_id: str, gpu: int, cfg: dict) -> bool:
    """Run pre-training script if configured. Returns True if OK to proceed."""
    import sys
    pre_script = cfg.get("pre_script")
    if not pre_script:
        return True

    python = cfg.get("python", sys.executable)
    pre_args = cfg.get("pre_args") or []
    pre_timeout = cfg.get("pre_timeout", 3600)

    from orze.engine.launcher import _format_args
    cmd = [python, pre_script]
    cmd.extend(_format_args(pre_args, {"idea_id": idea_id, "gpu": gpu}))

    env = os.environ.copy()
    for k, v in (cfg.get("train_extra_env") or {}).items():
        env[k] = str(v)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    logger.info("Running pre-script for %s on GPU %d", idea_id, gpu)
    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=pre_timeout,
        )
        if result.returncode == 0:
            logger.info("Pre-script OK for %s", idea_id)
            return True
        else:
            logger.warning("Pre-script failed for %s (exit %d): %s",
                           idea_id, result.returncode,
                           result.stderr[-200:] if result.stderr else "")
            return False
    except subprocess.TimeoutExpired:
        logger.warning("Pre-script timed out for %s after %ds",
                       idea_id, pre_timeout)
        return False
    except Exception as e:
        logger.warning("Pre-script error for %s: %s", idea_id, e)
        return False


def _new_process_group():
    """preexec_fn for subprocess.Popen to create a new process group."""
    os.setpgrp()
