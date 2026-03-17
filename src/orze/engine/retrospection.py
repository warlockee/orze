"""Periodic retrospection script runner.

CALLING SPEC:
    run_retrospection(results_dir, cfg, completed_count, last_count) -> int
        results_dir:      Path to results directory
        cfg:              orze.yaml config dict (needs 'retrospection' key)
        completed_count:  current number of completed experiments
        last_count:       last completed count when retrospection ran
        returns:          updated last_count (same as input if not triggered)

    Triggers when completed_count >= last_count + interval.
    Runs cfg['retrospection']['script'] as a subprocess.
    Writes output to results_dir / '_retrospection.txt'.
"""

import logging
import os
import subprocess
import sys
from pathlib import Path

logger = logging.getLogger("orze")


def run_retrospection(results_dir: Path, cfg: dict,
                      completed_count: int, last_count: int) -> int:
    """Run retrospection script if interval threshold crossed.

    Returns updated last_count (unchanged if not triggered).
    """
    retro_cfg = cfg.get("retrospection", {})
    if not retro_cfg.get("enabled") or not retro_cfg.get("script"):
        return last_count

    interval = retro_cfg.get("interval", 50)
    if completed_count < last_count + interval:
        return last_count

    logger.info("Retrospection triggered: %d completed (last run at %d, interval %d)",
                completed_count, last_count, interval)

    script = retro_cfg["script"]
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
                logger.info("Retrospection output written to %s", output_file)
            return completed_count
        else:
            logger.warning("Retrospection script failed (rc=%d): %s",
                           result.returncode, result.stderr[:300])
    except subprocess.TimeoutExpired:
        logger.warning("Retrospection script timed out after %ds", timeout)
    except Exception as e:
        logger.warning("Retrospection error: %s", e)

    return completed_count
