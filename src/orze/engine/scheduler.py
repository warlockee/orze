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
                  skipped: Optional[set] = None) -> List[str]:
    """Return idea IDs with no results dir, sorted by priority then ID.
    Excludes ideas in the skipped set."""
    unclaimed = []
    for idea_id in ideas:
        if skipped and idea_id in skipped:
            continue
        if not (results_dir / idea_id).exists():
            unclaimed.append(idea_id)

    def sort_key(idea_id):
        pri = PRIORITY_ORDER.get(ideas[idea_id]["priority"], 2)
        return (pri, idea_id)

    unclaimed.sort(key=sort_key)
    return unclaimed


# ---------------------------------------------------------------------------
# Claiming (atomic mkdir)
# ---------------------------------------------------------------------------

def claim(idea_id: str, results_dir: Path, gpu: int,
          lake=None) -> bool:
    """Atomically claim an idea via mkdir. Returns True if we got it.
    If lake is provided, also updates the DB status to 'running'."""
    idea_dir = results_dir / idea_id
    try:
        idea_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
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
    """Remove result dirs with claim.json but no metrics.json older than hours.
    If lake is provided, resets their DB status to 'queued' so they retry.
    Returns count of cleaned dirs."""
    if hours <= 0:
        return 0

    cleaned = 0
    cutoff = time.time() - hours * 3600

    for d in results_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("idea-"):
            continue
        claim_path = d / "claim.json"
        metrics_path = d / "metrics.json"

        if claim_path.exists() and not metrics_path.exists():
            try:
                last_activity = claim_path.stat().st_mtime
                log_path = d / "train_output.log"
                if log_path.exists():
                    last_activity = max(last_activity,
                                        log_path.stat().st_mtime)
                if last_activity < cutoff:
                    idea_id = d.name
                    shutil.rmtree(d)
                    logger.info("Cleaned orphan: %s (last activity %.1fh ago)",
                                idea_id, (time.time() - last_activity) / 3600)
                    if lake:
                        try:
                            lake.set_status(idea_id, "queued")
                        except Exception:
                            pass
                    cleaned += 1
            except Exception as e:
                logger.warning("Failed to clean orphan %s: %s", d.name, e)

    return cleaned


# ---------------------------------------------------------------------------
# Status counting
# ---------------------------------------------------------------------------

def _count_statuses(ideas: Dict[str, dict], results_dir: Path) -> dict:
    """Count idea statuses without full report generation."""
    counts = {}
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
            ideas_path = Path(cfg.get("ideas_file", "ideas.md"))
            lake_path = ideas_path.parent / "idea_lake.db"
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
