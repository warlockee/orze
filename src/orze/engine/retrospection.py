"""Periodic retrospection with auto-pause for research roles.

CALLING SPEC:
    run_retrospection(results_dir, cfg, completed_count, last_count) -> int
        results_dir:      Path to results directory
        cfg:              orze.yaml config dict (needs 'retrospection' key)
        completed_count:  current number of completed experiments
        last_count:       last completed count when retrospection ran
        returns:          updated last_count (same as input if not triggered)

    Triggers when completed_count >= last_count + interval.
    Runs cfg['retrospection']['script'] as a subprocess (if configured).
    Runs built-in plateau and failure-rate detection.
    Writes output to results_dir / '_retrospection.txt'.
    Writes results_dir / '.pause_research' sentinel to auto-pause research
    roles when a plateau or high failure rate is detected.

    is_research_paused(results_dir) -> bool
        Returns True if .pause_research sentinel exists.

    resume_research(results_dir) -> bool
        Removes .pause_research sentinel. Returns True if it existed.
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

logger = logging.getLogger("orze")

PAUSE_SENTINEL = ".pause_research"


def is_research_paused(results_dir: Path) -> bool:
    """Check if research is paused by retrospection."""
    return (results_dir / PAUSE_SENTINEL).exists()


def resume_research(results_dir: Path) -> bool:
    """Remove the pause sentinel to resume research. Returns True if it existed."""
    sentinel = results_dir / PAUSE_SENTINEL
    if sentinel.exists():
        sentinel.unlink()
        logger.info("Research resumed (pause sentinel removed)")
        return True
    return False


def _detect_plateau(results_dir: Path, window: int = 200) -> tuple:
    """Check if the best primary metric has improved in the last N completions.

    Returns (is_plateau: bool, message: str).
    """
    entries = []
    for d in results_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("idea-"):
            continue
        mf = d / "metrics.json"
        if not mf.exists():
            continue
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
            if m.get("status") == "COMPLETED":
                # Use avg_wer as primary metric; fall back to any numeric primary
                val = m.get("avg_wer") or m.get("test_accuracy") or m.get("score")
                if val is not None and isinstance(val, (int, float)) and val > 0:
                    entries.append((mf.stat().st_mtime, float(val)))
        except Exception:
            continue

    if len(entries) < window:
        return False, f"Not enough data ({len(entries)}/{window})"

    entries.sort(reverse=True)  # newest first
    recent = entries[:window]
    older = entries[window:]

    best_recent = min(v for _, v in recent)
    best_older = min(v for _, v in older) if older else float("inf")

    improved = best_recent < best_older
    return (
        not improved,
        f"best_recent={best_recent:.4f} best_older={best_older:.4f} improved={improved}",
    )


def _detect_high_failure_rate(results_dir: Path, window: int = 100,
                               threshold: float = 0.5) -> tuple:
    """Check if recent failure rate exceeds threshold.

    Returns (is_high: bool, message: str).
    """
    entries = []
    for d in results_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("idea-"):
            continue
        mf = d / "metrics.json"
        if not mf.exists():
            continue
        try:
            m = json.loads(mf.read_text(encoding="utf-8"))
            entries.append((mf.stat().st_mtime, m.get("status", "UNKNOWN")))
        except Exception:
            continue

    entries.sort(reverse=True)
    recent = entries[:window]
    if len(recent) < window // 2:
        return False, f"Not enough data ({len(recent)}/{window})"

    failures = sum(1 for _, s in recent if s == "FAILED")
    rate = failures / len(recent)
    return rate > threshold, f"recent_fail_rate={rate:.1%} ({failures}/{len(recent)})"


def _run_builtin_checks(results_dir: Path, cfg: dict,
                        completed_count: int) -> Optional[str]:
    """Run built-in plateau and failure-rate detection.

    Returns pause reason string if research should be paused, None otherwise.
    """
    retro_cfg = cfg.get("retrospection", {})
    plateau_window = retro_cfg.get("plateau_window", 200)
    fail_window = retro_cfg.get("fail_window", 100)
    fail_threshold = retro_cfg.get("fail_threshold", 0.5)

    reasons = []

    is_plateau, plateau_msg = _detect_plateau(results_dir, plateau_window)
    if is_plateau:
        reasons.append(f"PLATEAU: No improvement in last {plateau_window} experiments ({plateau_msg})")
        logger.warning("Retrospection: %s", reasons[-1])

    is_high_fail, fail_msg = _detect_high_failure_rate(
        results_dir, fail_window, fail_threshold)
    if is_high_fail:
        reasons.append(f"HIGH FAILURE RATE: {fail_msg}")
        logger.warning("Retrospection: %s", reasons[-1])

    if reasons:
        return "; ".join(reasons)
    return None


def run_retrospection(results_dir: Path, cfg: dict,
                      completed_count: int, last_count: int) -> int:
    """Run retrospection if interval threshold crossed.

    Runs the user's custom script (if configured) AND built-in checks.
    If a plateau or high failure rate is detected, writes a .pause_research
    sentinel that research roles should check before generating ideas.

    Returns updated last_count (unchanged if not triggered).
    """
    retro_cfg = cfg.get("retrospection", {})
    if not retro_cfg.get("enabled"):
        return last_count

    interval = retro_cfg.get("interval", 50)
    if completed_count < last_count + interval:
        return last_count

    logger.info("Retrospection triggered: %d completed (last run at %d, interval %d)",
                completed_count, last_count, interval)

    # 1. Run custom script (if configured)
    script = retro_cfg.get("script")
    if script:
        timeout = retro_cfg.get("timeout", 120)
        try:
            env = os.environ.copy()
            env["ORZE_RESULTS_DIR"] = str(results_dir)
            env["ORZE_COMPLETED_COUNT"] = str(completed_count)
            result = subprocess.run(
                [cfg.get("python", sys.executable), script],
                capture_output=True, text=True, timeout=timeout, env=env,
            )
            if result.returncode == 0:
                output_file = results_dir / "_retrospection.txt"
                if result.stdout.strip():
                    output_file.write_text(result.stdout, encoding="utf-8")
                    logger.info("Retrospection script output written to %s",
                                output_file)
            else:
                logger.warning("Retrospection script failed (rc=%d): %s",
                               result.returncode, result.stderr[:300])
        except subprocess.TimeoutExpired:
            logger.warning("Retrospection script timed out after %ds", timeout)
        except Exception as e:
            logger.warning("Retrospection script error: %s", e)

    # 2. Built-in plateau and failure-rate detection
    auto_pause = retro_cfg.get("auto_pause", True)
    if auto_pause:
        pause_reason = _run_builtin_checks(results_dir, cfg, completed_count)
        sentinel = results_dir / PAUSE_SENTINEL

        if pause_reason:
            sentinel.write_text(
                json.dumps({
                    "paused_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
                    "completed_count": completed_count,
                    "reason": pause_reason,
                }, indent=2),
                encoding="utf-8",
            )
            logger.warning("Research PAUSED by retrospection: %s", pause_reason)
        else:
            # Clear sentinel if conditions improved (e.g., manual ideas added)
            if sentinel.exists():
                sentinel.unlink()
                logger.info("Research RESUMED — plateau/failure conditions cleared")

    return completed_count
