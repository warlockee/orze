"""Training subprocess launcher and lifecycle monitor.

CALLING SPEC:
    launch(idea_id, gpu, results_dir, cfg) -> TrainingProcess
        idea_id: str — experiment identifier (e.g. "idea-abc123")
        gpu: int — CUDA device index (set as CUDA_VISIBLE_DEVICES)
        results_dir: Path — parent dir; logs written to results_dir/idea_id/train_output.log
        cfg: dict — orze config; requires keys 'train_script', 'ideas_file', 'base_config';
                     optional 'python', 'train_extra_args', 'train_extra_env', 'timeout'
        returns: TrainingProcess with a running Popen in its own process group
        side effects: creates results_dir/idea_id/train_output.log, spawns subprocess

    check_active(active, results_dir, cfg, failure_counts, fix_counts=None) -> list[(idea_id, gpu)]
        active: Dict[int, TrainingProcess] — gpu -> running process; MUTATED in-place (finished entries removed)
        results_dir: Path
        cfg: dict — uses 'stall_minutes', 'max_fix_attempts', executor config
        failure_counts: dict — idea_id -> int; MUTATED to track consecutive failures
        fix_counts: dict | None — idea_id -> int; MUTATED to track fix attempts
        returns: list of (idea_id, gpu) tuples for processes that finished this cycle
        side effects: kills timed-out/stalled/hung processes, writes metrics.json for failures,
                      may invoke executor LLM to auto-fix and relaunch failed ideas,
                      sends notifications on stall/timeout

    _format_args(args, template_vars) -> list[str]
        args: list | str | None — raw arguments (coerced to list)
        template_vars: dict — e.g. {"idea_id": "idea-abc", "gpu": 0}; replaces {key} in each arg
        returns: list of formatted string arguments

    _write_failure(idea_dir, reason) -> None
        idea_dir: Path — e.g. results_dir / idea_id
        reason: str — error description
        side effects: atomically writes {"status": "FAILED", "error": reason} to idea_dir/metrics.json

    _get_checkpoint_dir(cfg) -> Path | None
        cfg: dict — orze config
        returns: value of --checkpoint-dir from train_extra_args, or None
"""
import datetime
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from orze.engine.process import TrainingProcess, _new_process_group, _terminate_and_reap
from orze.core.fs import atomic_write, tail_file
from orze.reporting.notifications import notify

logger = logging.getLogger("orze")


def _get_checkpoint_dir(cfg: dict) -> Optional[Path]:
    """Extract --checkpoint-dir from train_extra_args."""
    args = cfg.get("train_extra_args") or []
    for i, arg in enumerate(args):
        if str(arg) == "--checkpoint-dir" and i + 1 < len(args):
            return Path(str(args[i + 1]))
    return None


def _format_args(args, template_vars: dict) -> list:
    """Safely format arguments without crashing on literal {} braces."""
    if args is None:
        args = []
    elif isinstance(args, str):
        args = [args]
    elif not isinstance(args, list):
        try:
            args = list(args)
        except TypeError:
            args = [args]
    formatted = []
    for arg in args:
        s = str(arg)
        for k, v in template_vars.items():
            s = s.replace(f"{{{k}}}", str(v))
        formatted.append(s)
    return formatted


def launch(idea_id: str, gpu: int, results_dir: Path, cfg: dict) -> TrainingProcess:
    """Launch a training subprocess on the given GPU."""
    log_path = results_dir / idea_id / "train_output.log"

    python = cfg.get("python", sys.executable)
    train_script = cfg["train_script"]

    cmd = [
        python, train_script,
        "--idea-id", idea_id,
        "--results-dir", str(results_dir),
        "--ideas-md", cfg["ideas_file"],
        "--config", cfg["base_config"],
    ]
    for arg in (cfg.get("train_extra_args") or []):
        cmd.append(str(arg))

    env = os.environ.copy()
    for k, v in (cfg.get("train_extra_env") or {}).items():
        env[k] = str(v)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Keep file handle open for subprocess lifetime
    log_fh = open(log_path, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
            preexec_fn=_new_process_group,
        )
    except Exception:
        log_fh.close()
        raise

    now = time.time()
    return TrainingProcess(
        idea_id=idea_id, gpu=gpu, process=proc,
        start_time=now, log_path=log_path,
        timeout=cfg.get("timeout", 3600),
        _log_fh=log_fh, _last_log_size=0,
        _last_log_check=now, _stall_since=0.0,
    )


def _write_failure(idea_dir: Path, reason: str):
    """Write a failure metrics.json atomically."""
    metrics = {
        "status": "FAILED",
        "error": reason,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    atomic_write(idea_dir / "metrics.json", json.dumps(metrics, indent=2))


def check_active(active: Dict[int, TrainingProcess], results_dir: Path,
                 cfg: dict, failure_counts: dict,
                 fix_counts: Optional[dict] = None) -> list:
    """Check running processes. Reap completed/timed-out/stalled/OOM.
    Returns list of (idea_id, gpu) tuples for finished ideas.

    When fix_counts is provided and max_fix_attempts > 0, failed ideas
    are sent to the executor LLM for diagnosis before recording failure.
    If the LLM applies a fix, the idea is re-launched on the same GPU.
    """
    from orze.engine.health import check_stalled, detect_fatal_in_log, _adaptive_stall_minutes
    from orze.engine.failure import _record_failure, _try_executor_fix, _reset_idea_for_retry

    finished = []
    stall_minutes = _adaptive_stall_minutes(
        results_dir, cfg.get("stall_minutes", 0))
    if fix_counts is None:
        fix_counts = {}

    for gpu in list(active.keys()):
        tp = active[gpu]
        ret = tp.process.poll()
        elapsed = time.time() - tp.start_time

        # --- Still running ---
        if ret is None:
            if elapsed > tp.timeout:
                logger.warning("[TIMEOUT] %s after %.0fm — killing",
                               tp.idea_id, elapsed / 60)
                notify("stall", {"idea_id": tp.idea_id, "gpu": gpu,
                                 "reason": f"Timeout after {elapsed / 60:.0f}m"}, cfg)
                _terminate_and_reap(tp.process, tp.idea_id)
                tp.close_log()
                error_msg = "Timed out"
                if _try_executor_fix(tp.idea_id, error_msg,
                                     results_dir, cfg, fix_counts):
                    _reset_idea_for_retry(results_dir / tp.idea_id)
                    try:
                        new_tp = launch(tp.idea_id, gpu, results_dir, cfg)
                        active[gpu] = new_tp
                        logger.info("[FIX-RETRY] %s relaunched on GPU %d",
                                     tp.idea_id, gpu)
                        continue
                    except Exception as e:
                        logger.error("[FIX-RETRY] %s relaunch failed: %s",
                                      tp.idea_id, e)
                _write_failure(results_dir / tp.idea_id, error_msg)
                _record_failure(failure_counts, tp.idea_id)
                del active[gpu]
                finished.append((tp.idea_id, gpu))
                continue

            if check_stalled(tp, stall_minutes):
                logger.warning("[STALLED] %s — no log output for %dm, killing",
                               tp.idea_id, stall_minutes)
                notify("stall", {"idea_id": tp.idea_id, "gpu": gpu,
                                 "reason": f"Stalled ({stall_minutes}m no output)"}, cfg)
                _terminate_and_reap(tp.process, tp.idea_id)
                tp.close_log()
                error_msg = f"Stalled (no output for {stall_minutes}m)"
                if _try_executor_fix(tp.idea_id, error_msg,
                                     results_dir, cfg, fix_counts):
                    _reset_idea_for_retry(results_dir / tp.idea_id)
                    try:
                        new_tp = launch(tp.idea_id, gpu, results_dir, cfg)
                        active[gpu] = new_tp
                        logger.info("[FIX-RETRY] %s relaunched on GPU %d",
                                     tp.idea_id, gpu)
                        continue
                    except Exception as e:
                        logger.error("[FIX-RETRY] %s relaunch failed: %s",
                                      tp.idea_id, e)
                _write_failure(results_dir / tp.idea_id, error_msg)
                _record_failure(failure_counts, tp.idea_id)
                del active[gpu]
                finished.append((tp.idea_id, gpu))
                continue

            fatal = detect_fatal_in_log(tp)
            if fatal and tp.process.poll() is None:
                logger.warning("[FATAL-HUNG] %s — fatal error in log but "
                               "process still alive, killing:\n%s",
                               tp.idea_id, fatal[:200])
                notify("stall", {"idea_id": tp.idea_id, "gpu": gpu,
                                 "reason": f"Fatal error (hung): {fatal[:100]}"}, cfg)
                _terminate_and_reap(tp.process, tp.idea_id)
                tp.close_log()
                error_msg = f"Process hung after fatal error:\n{fatal[:500]}"
                if _try_executor_fix(tp.idea_id, error_msg,
                                     results_dir, cfg, fix_counts):
                    _reset_idea_for_retry(results_dir / tp.idea_id)
                    try:
                        new_tp = launch(tp.idea_id, gpu, results_dir, cfg)
                        active[gpu] = new_tp
                        logger.info("[FIX-RETRY] %s relaunched on GPU %d",
                                     tp.idea_id, gpu)
                        continue
                    except Exception as e:
                        logger.error("[FIX-RETRY] %s relaunch failed: %s",
                                      tp.idea_id, e)
                _write_failure(results_dir / tp.idea_id, error_msg)
                _record_failure(failure_counts, tp.idea_id)
                del active[gpu]
                finished.append((tp.idea_id, gpu))
                continue

            kill_file = results_dir / tp.idea_id / ".kill"
            if kill_file.exists():
                logger.info("Admin kill signal for %s — terminating", tp.idea_id)
                _terminate_and_reap(tp.process)
                tp.close_log()
                kill_file.unlink(missing_ok=True)
                _write_failure(results_dir / tp.idea_id, "Killed by admin")
                del active[gpu]
                finished.append((tp.idea_id, gpu))

            continue

        # --- Process exited ---
        # Reap zombie to prevent accumulation
        try:
            tp.process.wait(timeout=1)
        except Exception:
            pass
        tp.close_log()
        metrics_path = results_dir / tp.idea_id / "metrics.json"

        if ret == 0 and metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                metrics = {"status": "UNKNOWN"}
            status = metrics.get("status", "COMPLETED")
            logger.info("[%s] %s on GPU %d in %.1fm",
                        status, tp.idea_id, gpu, elapsed / 60)
            if status == "FAILED":
                error_msg = metrics.get("error", "Training script reported FAILED")
                if _try_executor_fix(tp.idea_id, error_msg,
                                     results_dir, cfg, fix_counts):
                    _reset_idea_for_retry(results_dir / tp.idea_id)
                    try:
                        new_tp = launch(tp.idea_id, gpu, results_dir, cfg)
                        active[gpu] = new_tp
                        logger.info("[FIX-RETRY] %s relaunched on GPU %d",
                                     tp.idea_id, gpu)
                        continue
                    except Exception as e:
                        logger.error("[FIX-RETRY] %s relaunch failed: %s",
                                      tp.idea_id, e)
                _record_failure(failure_counts, tp.idea_id)
        else:
            reason = f"exit code {ret}"
            try:
                tail_str = tail_file(tp.log_path, 8192)
                lines = tail_str.strip().split("\n")
                tail = "\n".join(lines[-5:])
                reason += f"\n{tail}"
            except Exception:
                pass
            logger.warning("[FAILED] %s on GPU %d — %s", tp.idea_id, gpu, reason)
            if _try_executor_fix(tp.idea_id, reason,
                                 results_dir, cfg, fix_counts):
                _reset_idea_for_retry(results_dir / tp.idea_id)
                try:
                    new_tp = launch(tp.idea_id, gpu, results_dir, cfg)
                    active[gpu] = new_tp
                    logger.info("[FIX-RETRY] %s relaunched on GPU %d",
                                 tp.idea_id, gpu)
                    continue
                except Exception as e:
                    logger.error("[FIX-RETRY] %s relaunch failed: %s",
                                  tp.idea_id, e)
            if not metrics_path.exists():
                _write_failure(results_dir / tp.idea_id,
                               f"Process exited with code {ret}")
            _record_failure(failure_counts, tp.idea_id)

        del active[gpu]
        finished.append((tp.idea_id, gpu))

    return finished
