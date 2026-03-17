"""Post-training evaluation subprocess launcher and monitor.

CALLING SPEC:
    launch_eval(idea_id, gpu, results_dir, cfg) -> EvalProcess | None
        idea_id: str — experiment identifier
        gpu: int — CUDA device index (NOT set as CUDA_VISIBLE_DEVICES; eval script uses --gpu)
        results_dir: Path — parent dir for experiment results
        cfg: dict — requires 'eval_script'; optional 'eval_args', 'eval_timeout', 'eval_output',
                     'python', 'train_extra_env'
        returns: EvalProcess if launched, None if eval_script missing, already evaluated,
                 or training status != COMPLETED
        side effects: spawns subprocess, creates results_dir/idea_id/eval_output.log

    check_active_evals(active_evals, results_dir, cfg) -> list[(idea_id, gpu)]
        active_evals: Dict[int, EvalProcess] — gpu -> running eval; MUTATED in-place (finished entries removed)
        results_dir: Path
        cfg: dict — uses 'eval_output' (default "eval_report.json")
        returns: list of (idea_id, gpu) tuples for evals that finished this cycle
        side effects: kills timed-out evals, writes failure marker JSON if eval dies without output

    run_eval(idea_id, gpu, results_dir, cfg) -> None
        Blocking wrapper around launch_eval; waits for completion.
        Used in --once mode. Writes failure marker on error/timeout.
        side effects: same as launch_eval, but blocks until done

    run_post_scripts(idea_id, gpu, results_dir, cfg) -> None
        idea_id: str
        gpu: int — set as CUDA_VISIBLE_DEVICES for post-scripts
        results_dir: Path
        cfg: dict — uses 'post_scripts' (list of {script, args, timeout, output, name}),
                     'python', 'train_extra_env'
        side effects: runs each post-script sequentially (blocking), skips if output file exists,
                      skips entirely if training status != COMPLETED
"""
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, Optional

from orze.engine.process import EvalProcess, _new_process_group, _terminate_and_reap
from orze.engine.launcher import _format_args
from orze.core.fs import tail_file

logger = logging.getLogger("orze")


def launch_eval(idea_id: str, gpu: int, results_dir: Path,
                cfg: dict) -> Optional[EvalProcess]:
    """Launch a non-blocking eval subprocess. Returns EvalProcess or None."""
    eval_script = cfg.get("eval_script")
    if not eval_script:
        return None

    eval_output = cfg.get("eval_output") or "eval_report.json"
    output_path = results_dir / idea_id / eval_output
    if output_path.exists():
        logger.debug("Eval already exists for %s, skipping", idea_id)
        return None

    metrics_path = results_dir / idea_id / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics.get("status") != "COMPLETED":
                return None
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None
    else:
        return None

    python = cfg.get("python", sys.executable)
    eval_args = cfg.get("eval_args") or []
    eval_timeout = cfg.get("eval_timeout", 3600)

    cmd = [python, eval_script]
    cmd.extend(_format_args(eval_args, {"idea_id": idea_id, "gpu": gpu}))

    log_path = results_dir / idea_id / "eval_output.log"
    logger.info("Launching eval for %s on GPU %d", idea_id, gpu)

    try:
        env = os.environ.copy()
        for k, v in (cfg.get("train_extra_env") or {}).items():
            env[k] = str(v)
        # Don't set CUDA_VISIBLE_DEVICES — the eval script uses --gpu
        # to select the device via torch.cuda.set_device().
        env.pop("CUDA_VISIBLE_DEVICES", None)
        log_fh = open(log_path, "w", encoding="utf-8")
        try:
            proc = subprocess.Popen(
                cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
                preexec_fn=_new_process_group,
            )
        except Exception:
            log_fh.close()
            raise
        return EvalProcess(
            idea_id=idea_id, gpu=gpu, process=proc,
            start_time=time.time(), log_path=log_path,
            timeout=eval_timeout, _log_fh=log_fh,
        )
    except Exception as e:
        logger.warning("Failed to launch eval for %s: %s", idea_id, e)
        return None


def run_eval(idea_id: str, gpu: int, results_dir: Path, cfg: dict):
    """Run post-training evaluation (blocking). Used in --once mode."""
    ep = launch_eval(idea_id, gpu, results_dir, cfg)
    if ep is None:
        return
    eval_output = cfg.get("eval_output") or "eval_report.json"
    reason = ""
    try:
        ep.process.wait(timeout=ep.timeout)
        if ep.process.returncode == 0:
            logger.info("Eval completed for %s", idea_id)
        else:
            reason = f"Exit code {ep.process.returncode}"
            logger.warning("Eval failed for %s (exit %d)",
                           idea_id, ep.process.returncode)
    except subprocess.TimeoutExpired:
        reason = f"Timed out after {ep.timeout}s"
        logger.warning("Eval timed out for %s after %ds",
                       idea_id, ep.timeout)
        _terminate_and_reap(ep.process, f"eval {idea_id}")
    except Exception as e:
        reason = str(e)
        logger.warning("Eval error for %s: %s", idea_id, e)
    finally:
        ep.close_log()
        if reason:
            _write_eval_failure_marker(results_dir, idea_id, eval_output, reason)


def _write_eval_failure_marker(results_dir: Path, idea_id: str,
                               eval_output: str, reason: str) -> None:
    """Safety net: write failure marker if eval process died without one.

    The marker file is the eval_output itself so the backlog scanner
    won't re-queue this idea.  The eval script is responsible for
    writing domain-specific reports; this is a generic fallback.
    """
    report_path = results_dir / idea_id / eval_output
    if report_path.exists():
        return  # Script already wrote a report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "status": "FAILED",
        "reason": reason[:500],
    }, indent=2))
    logger.info("Wrote eval failure marker for %s", idea_id)


def check_active_evals(active_evals: Dict[int, EvalProcess],
                       results_dir: Path, cfg: dict) -> list:
    """Check running eval processes. Returns list of (idea_id, gpu) for finished evals."""
    eval_output = cfg.get("eval_output") or "eval_report.json"
    finished = []
    for gpu in list(active_evals.keys()):
        ep = active_evals[gpu]
        ret = ep.process.poll()
        elapsed = time.time() - ep.start_time

        if ret is None:
            # Still running — check timeout
            if elapsed > ep.timeout:
                logger.warning("[EVAL TIMEOUT] %s after %.0fm — killing",
                               ep.idea_id, elapsed / 60)
                _terminate_and_reap(ep.process, f"eval {ep.idea_id}")
                ep.close_log()
                _write_eval_failure_marker(
                    results_dir, ep.idea_id, eval_output,
                    f"Timed out after {elapsed/60:.0f}m")
                del active_evals[gpu]
                finished.append((ep.idea_id, gpu))
            continue

        # Process exited
        ep.close_log()
        if ret == 0:
            logger.info("[EVAL OK] %s on GPU %d in %.1fm",
                        ep.idea_id, gpu, elapsed / 60)
        else:
            # Log tail of eval output for diagnosis
            eval_tail = tail_file(ep.log_path, 2048).strip()
            logger.warning("[EVAL FAILED] %s on GPU %d — exit %d\n%s",
                           ep.idea_id, gpu, ret,
                           eval_tail[-500:] if eval_tail else "(no output)")
            _write_eval_failure_marker(
                results_dir, ep.idea_id, eval_output,
                f"Exit code {ret}: {eval_tail[-300:] if eval_tail else 'no output'}")
        del active_evals[gpu]
        finished.append((ep.idea_id, gpu))

    return finished


def run_post_scripts(idea_id: str, gpu: int, results_dir: Path, cfg: dict):
    """Run additional post-training scripts (beyond eval_script).
    Each entry in post_scripts is a dict with: script, args, timeout, output."""
    post_scripts = (cfg.get("post_scripts") or [])
    if not post_scripts:
        return

    # Check training succeeded
    metrics_path = results_dir / idea_id / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics.get("status") != "COMPLETED":
                return
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return
    else:
        return

    python = cfg.get("python", sys.executable)
    env = os.environ.copy()
    for k, v in (cfg.get("train_extra_env") or {}).items():
        env[k] = str(v)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    for i, ps in enumerate(post_scripts):
        script = ps.get("script")
        if not script:
            continue

        # Skip if output already exists
        output_file = ps.get("output", "")
        if output_file:
            output_path = results_dir / idea_id / output_file
            if output_path.exists():
                logger.debug("Post-script %d output exists for %s, skipping",
                             i, idea_id)
                continue

        args = ps.get("args") or []
        timeout = ps.get("timeout", 3600)
        name = ps.get("name", f"post-script-{i}")

        cmd = [python, script]
        cmd.extend(_format_args(args, {"idea_id": idea_id, "gpu": gpu}))

        log_path = results_dir / idea_id / f"{name}.log"
        logger.info("Running %s for %s", name, idea_id)

        try:
            with open(log_path, "w", encoding="utf-8") as log_fh:
                result = subprocess.run(
                    cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            if result.returncode == 0:
                logger.info("%s completed for %s", name, idea_id)
            else:
                logger.warning("%s failed for %s (exit %d)",
                               name, idea_id, result.returncode)
        except subprocess.TimeoutExpired:
            logger.warning("%s timed out for %s after %ds",
                           name, idea_id, timeout)
        except Exception as e:
            logger.warning("%s error for %s: %s", name, idea_id, e)
