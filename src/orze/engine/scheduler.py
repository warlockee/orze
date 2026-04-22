"""Idea scheduling, claiming, orphan cleanup, and status counting.

CALLING SPEC:
    get_unclaimed(ideas, results_dir, skipped=None) -> list[str]
        ideas: Dict[str, dict] — parsed ideas (id -> metadata with 'priority')
        results_dir: Path — parent dir for experiment results
        skipped: set | None — idea IDs to exclude
        returns: idea IDs that have no results_dir/idea_id directory,
                 sorted by priority (critical > high > medium > low) then ID

    claim(idea_id, results_dir, gpu, lake=None) -> bool
        idea_id: str
        results_dir: Path
        gpu: int — recorded in claim.json
        lake: IdeaLake | None — if provided, sets status to 'running'
        returns: True if mkdir succeeded (we got the lock), False if already claimed
        side effects: creates results_dir/idea_id/ directory, writes claim.json

    cleanup_orphans(results_dir, hours, lake=None) -> int
        results_dir: Path
        hours: float — max age of stale claims (0 disables cleanup)
        lake: IdeaLake | None — if provided, resets cleaned ideas to 'queued'
        returns: number of orphan directories removed
        side effects: deletes results_dir/idea-*/  dirs that have claim.json but
                      no metrics.json and no activity for > hours

    _count_statuses(ideas, results_dir) -> dict
        ideas: Dict[str, dict]
        results_dir: Path
        returns: {"QUEUED": n, "IN_PROGRESS": n, "COMPLETED": n, "FAILED": n, ...}

    run_cleanup(results_dir, cfg) -> None
        results_dir: Path
        cfg: dict — uses 'cleanup' (patterns, script, timeout), 'gc' (enabled,
                     checkpoints_dir, keep_top, keep_recent, min_free_gb), 'report', 'ideas_file'
        side effects: deletes checkpoint dirs via GC, deletes files matching glob patterns
                      in results dirs, runs custom cleanup script
"""
import os
import re
import shutil
import subprocess
import time
import socket
import datetime
import json
import logging
import sys
from typing import Dict, List, Optional
from pathlib import Path
from orze.core.fs import _fs_lock, _fs_unlock, atomic_write

logger = logging.getLogger("orze")
PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def get_unclaimed(ideas: Dict[str, dict], results_dir: Path,
                  skipped: Optional[set] = None,
                  lake=None) -> List[str]:
    """Return idea IDs with no results dir, sorted by priority then ID.
    Excludes ideas in the skipped set and ideas with missing strategy files.
    When lake is provided, permanently marks unfixable ideas as 'skipped'
    so they don't clog the queue forever."""
    unclaimed = []
    for idea_id in ideas:
        if skipped and idea_id in skipped:
            continue
        if not (results_dir / idea_id).exists():
            # Validate strategy file exists before counting as queued
            idea_config = ideas[idea_id].get("config", {})
            strategy_name = idea_config.get("strategy")
            if strategy_name:
                strategy_path = Path("strategies") / f"{strategy_name}.py"
                if not strategy_path.exists():
                    logger.warning(
                        "Skipping %s: strategy file %s does not exist",
                        idea_id, strategy_path)
                    if lake:
                        try:
                            lake.set_status(idea_id, "skipped")
                        except Exception:
                            pass
                    continue
            unclaimed.append(idea_id)

    def sort_key(idea_id):
        pri = PRIORITY_ORDER.get(ideas[idea_id]["priority"], 2)
        # Inference-only ideas run in minutes, training ideas run in hours.
        # When all GPU slots are held by long training runs, inference-only
        # ideas can starve indefinitely (observed: a domain_lora_routing eval
        # queued "P1" in retrospection for 10+ cycles, never dispatched).
        # Boost them one tier within their priority bucket so a freed GPU
        # picks the cheap eval before the next heavy train.
        #
        # Signals are conservative so projects without a matching
        # convention aren't silently reordered:
        #   1. explicit config flag inference_only: true
        #   2. strategy name == "eval_only" or ending in "_eval"
        # Other projects can opt in by setting inference_only: true on
        # ideas that only load a checkpoint and evaluate.
        cfg = ideas[idea_id].get("config") or {}
        strategy = str(cfg.get("strategy") or "").lower()
        is_inference_only = (
            bool(cfg.get("inference_only"))
            or strategy == "eval_only"
            or strategy.endswith("_eval"))
        inference_boost = 0 if is_inference_only else 1
        return (pri, inference_boost, idea_id)

    unclaimed.sort(key=sort_key)
    return unclaimed


# ---------------------------------------------------------------------------
# Claiming (atomic mkdir)
# ---------------------------------------------------------------------------

def claim(idea_id: str, results_dir: Path, gpu: int,
          lake=None) -> bool:
    """Atomically claim an idea via mkdir. Returns True if we got it.
    If lake is provided, also updates the DB status to 'running'.

    Multi-host safety: if the directory already exists with a claim.json
    owned by another host, refuse to claim (even if the idea was reset
    to 'queued' in the DB by a stale reconciliation).
    """
    idea_dir = results_dir / idea_id
    try:
        idea_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        # Dir exists — check if another host owns it
        claim_path = idea_dir / "claim.json"
        if claim_path.exists():
            try:
                existing = json.loads(claim_path.read_text(encoding="utf-8"))
                if existing.get("claimed_by") != socket.gethostname():
                    return False  # another host owns this, don't steal
            except (json.JSONDecodeError, OSError):
                pass
        return False

    claim_info = {
        "claimed_by": socket.gethostname(),
        "claimed_at": datetime.datetime.now().isoformat(),
        "pid": os.getpid(),
        "gpu": gpu,
    }
    atomic_write(idea_dir / "claim.json", json.dumps(claim_info, indent=2))

    if lake:
        try:
            lake.set_status(idea_id, "running")
        except Exception:
            pass  # filesystem is the primary lock, DB is best-effort

    return True


# ---------------------------------------------------------------------------
# GPU management
# ---------------------------------------------------------------------------

def cleanup_orphans(results_dir: Path, hours: float,
                    lake=None) -> int:
    """Reclaim stale claims on result dirs. Returns count reclaimed.

    Two classes of stale claims are reclaimed:

    1. ``claim.json`` exists, ``metrics.json`` does NOT, activity older than
       ``hours``: the whole directory is deleted (no useful output) and the
       lake status is reset to 'queued' so the idea retries.

    2. ``claim.json`` exists, ``metrics.json`` DOES exist, but the lake
       still says ``status='running'`` and activity is older than ``hours``:
       the training crashed mid-run after writing partial metrics (e.g. a
       second node died between epochs). The dir is kept so partial
       checkpoints/logs remain inspectable, but ``claim.json`` is removed
       and lake status reset to 'queued' so the idea can be re-attempted.

    Before this second class was handled, any cross-host claim that had
    written partial metrics got stuck as 'running' forever, silently
    consuming a queue slot even after the owning host was confirmed dead.
    """
    if hours <= 0:
        return 0

    cleaned = 0
    cutoff = time.time() - hours * 3600

    for d in results_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("idea-"):
            continue
        claim_path = d / "claim.json"
        metrics_path = d / "metrics.json"

        if not claim_path.exists():
            continue

        try:
            last_activity = claim_path.stat().st_mtime
            log_path = d / "train_output.log"
            if log_path.exists():
                last_activity = max(last_activity,
                                    log_path.stat().st_mtime)
            if last_activity >= cutoff:
                continue

            idea_id = d.name
            age_hours = (time.time() - last_activity) / 3600

            if not metrics_path.exists():
                # No output at all — delete the whole dir.
                shutil.rmtree(d)
                logger.info("Cleaned orphan: %s (last activity %.1fh ago)",
                            idea_id, age_hours)
                if lake:
                    try:
                        lake.set_status(idea_id, "queued")
                    except Exception:
                        pass
                cleaned += 1
                continue

            # metrics.json exists. Only reclaim if lake still thinks this
            # is running — i.e. training crashed before marking completion.
            # Finished ideas (completed/failed/skipped) stay untouched.
            if lake is None:
                continue
            try:
                idea = lake.get(idea_id)
            except Exception:
                continue
            if not idea or idea.get("status") != "running":
                continue

            try:
                claim_path.unlink()
            except OSError as e:
                logger.warning("Failed to unlink claim for %s: %s",
                               idea_id, e)
                continue
            try:
                lake.set_status(idea_id, "queued")
            except Exception as e:
                logger.warning("Failed to re-queue %s: %s", idea_id, e)
                continue
            logger.info(
                "Reclaimed stale running claim: %s (last activity %.1fh ago, "
                "partial metrics preserved)", idea_id, age_hours)
            cleaned += 1
        except Exception as e:
            logger.warning("Failed to clean orphan %s: %s", d.name, e)

    return cleaned


# ---------------------------------------------------------------------------
# Status counting
# ---------------------------------------------------------------------------

def _count_statuses(ideas: Dict[str, dict], results_dir: Path,
                    lake=None) -> dict:
    """Count idea statuses without full report generation.

    When a lake (IdeaLake) is provided, completed/failed counts are sourced
    from the database so they survive commander restarts.  The ``ideas`` dict
    typically contains only queued/in-progress items, which would make the
    completed count appear as zero after a restart.
    """
    counts = {}

    # Count ideas from the current queue (queued / in-progress)
    for idea_id in ideas:
        idea_dir = results_dir / idea_id
        if not idea_dir.exists():
            counts["QUEUED"] = counts.get("QUEUED", 0) + 1
        elif (idea_dir / "metrics.json").exists():
            try:
                m = json.loads((idea_dir / "metrics.json").read_text(encoding="utf-8"))
                st = m.get("status", "UNKNOWN")
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                st = "FAILED"
            counts[st] = counts.get(st, 0) + 1
        else:
            counts["IN_PROGRESS"] = counts.get("IN_PROGRESS", 0) + 1

    # Merge authoritative counts from the lake (includes archived ideas
    # that are no longer in the hot ideas dict).
    if lake is not None:
        try:
            rows = lake.conn.execute(
                "SELECT status, COUNT(*) FROM ideas GROUP BY status"
            ).fetchall()
            for status, cnt in rows:
                key = status.upper()
                if key in ("COMPLETED", "FAILED"):
                    # Lake is authoritative for completed/failed — override
                    counts[key] = cnt
                elif key == "QUEUED" and "QUEUED" not in counts:
                    counts["QUEUED"] = cnt
        except Exception:
            pass

    return counts


# ---------------------------------------------------------------------------
# Garbage collection / cleanup
# ---------------------------------------------------------------------------

def run_cleanup(results_dir: Path, cfg: dict):
    """Run periodic cleanup: GC checkpoints, delete file patterns, run script."""
    cleanup_cfg = cfg.get("cleanup") or {}

    # GC: delete checkpoint dirs for non-top experiments
    gc_cfg = cfg.get("gc") or {}
    if gc_cfg.get("enabled") and gc_cfg.get("checkpoints_dir"):
        try:
            from orze.agents.orze_gc import run_gc
            report_cfg = cfg.get("report") or {}
            lake_path = Path(cfg.get("idea_lake_db") or Path(cfg.get("results_dir", "orze_results")) / "idea_lake.db")
            stats = run_gc(
                results_dir=results_dir,
                checkpoints_dir=Path(gc_cfg["checkpoints_dir"]),
                primary_metric=report_cfg.get("primary_metric", ""),
                lake_db_path=lake_path if lake_path.exists() else None,
                keep_top=gc_cfg.get("keep_top", 50),
                keep_recent=gc_cfg.get("keep_recent", 20),
                min_free_gb=gc_cfg.get("min_free_gb", 0),
            )
            cs = stats.get("checkpoints", {})
            if cs.get("deleted", 0) > 0:
                logger.info("GC: deleted %d checkpoint dirs, kept %d",
                            cs["deleted"], cs["kept"])
        except Exception as e:
            logger.warning("GC failed: %s", e)

    # Built-in: delete files matching glob patterns in results dirs
    patterns = cleanup_cfg.get("patterns") or []
    if patterns:
        deleted = 0
        for d in results_dir.iterdir():
            if not d.is_dir() or not d.name.startswith("idea-"):
                continue
            for pattern in patterns:
                for f in d.glob(pattern):
                    try:
                        if f.is_file():
                            f.unlink()
                            deleted += 1
                    except Exception:
                        pass
        if deleted:
            logger.info("Cleanup: deleted %d files matching %s",
                        deleted, patterns)

    # Custom cleanup script
    script = cleanup_cfg.get("script")
    if script:
        python = cfg.get("python", sys.executable)
        timeout = cleanup_cfg.get("timeout", 300)
        try:
            result = subprocess.run(
                [python, script], capture_output=True, text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                logger.info("Cleanup script completed")
            else:
                logger.warning("Cleanup script failed (exit %d)",
                               result.returncode)
        except Exception as e:
            logger.warning("Cleanup script error: %s", e)


# ---------------------------------------------------------------------------
# Process checking (with health monitoring)
