#!/usr/bin/env python3
"""orze gc: generic garbage collector for experiment checkpoints and artifacts.

Frees disk space by deleting checkpoint directories for non-top experiments.
Determines which experiments to keep from _leaderboard.json and idea_lake.db.

Can be run standalone or wired into farm.py's cleanup cycle.

Usage:
    # Standalone
    python orze/orze_gc.py -c orze.yaml --dry-run
    python orze/orze_gc.py -c orze.yaml --keep-top 50

    # In orze.yaml (auto-called by farm.py every cleanup.interval iterations):
    gc:
      enabled: true
      checkpoints_dir: checkpoints      # where model checkpoints live
      keep_top: 50                       # keep top N by primary_metric
      keep_recent: 20                    # also keep N most recently completed
      min_free_gb: 100                   # only run GC when disk < this
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import sqlite3
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("orze.gc")


def get_top_idea_ids(results_dir: Path, primary_metric: str,
                     lake_db_path: Optional[Path], keep_top: int) -> Set[str]:
    """Collect idea IDs that should be kept (top performers).

    Sources (merged):
    1. _leaderboard.json (top 20 from farm.py)
    2. idea_lake.db (top N by primary_metric)
    """
    keep: Set[str] = set()

    # 1. Leaderboard (always trust — these are the current best)
    lb_path = results_dir / "_leaderboard.json"
    if lb_path.exists():
        try:
            data = json.loads(lb_path.read_text(encoding="utf-8"))
            for entry in data.get("top", []):
                iid = entry.get("idea_id", "")
                if iid:
                    keep.add(iid)
        except (json.JSONDecodeError, OSError):
            pass

    # 2. Idea Lake (broader history)
    if lake_db_path and lake_db_path.exists() and primary_metric:
        try:
            conn = sqlite3.connect(str(lake_db_path), timeout=10)
            # Try primary_metric, then fallback without "adjusted_"
            for metric in [primary_metric,
                           primary_metric.replace("_adjusted_", "_")]:
                rows = conn.execute(
                    "SELECT idea_id FROM ideas "
                    "WHERE json_extract(eval_metrics, ?) IS NOT NULL "
                    "ORDER BY json_extract(eval_metrics, ?) DESC LIMIT ?",
                    (f"$.{metric}", f"$.{metric}", keep_top),
                ).fetchall()
                if rows:
                    for r in rows:
                        keep.add(r[0])
                    break
            conn.close()
        except Exception as e:
            logger.warning("Lake query failed: %s", e)

    return keep


def get_recent_idea_ids(results_dir: Path, keep_recent: int) -> Set[str]:
    """Get the N most recently completed idea IDs (by metrics.json mtime)."""
    recent: List[tuple] = []  # (mtime, idea_id)

    try:
        with os.scandir(results_dir) as it:
            for entry in it:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                if not entry.name.startswith("idea-"):
                    continue
                metrics_path = Path(entry.path) / "metrics.json"
                if metrics_path.exists():
                    try:
                        data = json.loads(metrics_path.read_text("utf-8"))
                        if data.get("status") == "COMPLETED":
                            recent.append((metrics_path.stat().st_mtime,
                                           entry.name))
                    except (json.JSONDecodeError, OSError):
                        pass
    except OSError:
        pass

    recent.sort(reverse=True)
    return {iid for _, iid in recent[:keep_recent]}


def get_active_idea_ids(results_dir: Path) -> Set[str]:
    """Get idea IDs that are currently being trained (have claim but no metrics)."""
    active: Set[str] = set()
    try:
        with os.scandir(results_dir) as it:
            for entry in it:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                if not entry.name.startswith("idea-"):
                    continue
                claim = Path(entry.path) / "claim.json"
                metrics = Path(entry.path) / "metrics.json"
                if claim.exists() and not metrics.exists():
                    active.add(entry.name)
    except OSError:
        pass
    return active


def gc_checkpoints(checkpoints_dir: Path, keep_ids: Set[str],
                   dry_run: bool = False) -> Dict[str, Any]:
    """Delete checkpoint directories for ideas NOT in keep_ids.

    Returns stats: {deleted: int, freed_bytes: int, kept: int, errors: int}
    """
    stats = {"deleted": 0, "freed_bytes": 0, "kept": 0, "errors": 0}

    if not checkpoints_dir.exists():
        logger.info("Checkpoints dir does not exist: %s", checkpoints_dir)
        return stats

    # Scan all subdirectories (may be nested one level: checkpoints/subdir/idea-*)
    dirs_to_check: List[Path] = []
    for entry in os.scandir(checkpoints_dir):
        if not entry.is_dir(follow_symlinks=False):
            continue
        # Check if this is an idea dir directly
        if entry.name.startswith("idea-"):
            dirs_to_check.append(Path(entry.path))
        else:
            # Check one level deeper (e.g. checkpoints/project/idea-*)
            subdir = Path(entry.path)
            try:
                for sub_entry in os.scandir(subdir):
                    if sub_entry.is_dir(follow_symlinks=False) and \
                       sub_entry.name.startswith("idea-"):
                        dirs_to_check.append(Path(sub_entry.path))
            except OSError:
                pass

    for idea_dir in sorted(dirs_to_check):
        # Strip sweep suffix for keep check: idea-123~lr=0.001 → idea-123
        base_id = idea_dir.name.split("~")[0]

        if base_id in keep_ids:
            stats["kept"] += 1
            continue

        if dry_run:
            logger.info("[DRY RUN] Would delete: %s", idea_dir)
            stats["deleted"] += 1
            continue

        try:
            shutil.rmtree(idea_dir)
            stats["deleted"] += 1
        except Exception as e:
            logger.warning("Failed to delete %s: %s", idea_dir, e)
            stats["errors"] += 1

    return stats


def gc_results(results_dir: Path, keep_ids: Set[str],
               dry_run: bool = False) -> Dict[str, Any]:
    """Delete large artifacts (.pt, .pth) from results/ directories of non-top ideas.

    Unlike checkpoints_dir (which deletes the entire dir), this only prunes
    large files to keep logs and metrics intact.
    """
    stats = {"deleted_files": 0, "freed_bytes": 0, "kept": 0, "errors": 0}

    if not results_dir.exists():
        return stats

    try:
        with os.scandir(results_dir) as it:
            for entry in it:
                if not entry.is_dir(follow_symlinks=False):
                    continue
                if not entry.name.startswith("idea-"):
                    continue

                base_id = entry.name.split("~")[0]
                if base_id in keep_ids:
                    stats["kept"] += 1
                    continue

                # Prune large files in this results dir
                idea_path = Path(entry.path)
                for ext in ["*.pt", "*.pth", "*.ckpt", "*.bin"]:
                    for f in idea_path.glob(ext):
                        if dry_run:
                            logger.info("[DRY RUN] Would delete artifact: %s", f)
                            stats["deleted_files"] += 1
                            continue
                        try:
                            f_size = f.stat().st_size
                            f.unlink()
                            stats["deleted_files"] += 1
                            stats["freed_bytes"] += f_size
                        except Exception as e:
                            logger.warning("Failed to delete %s: %s", f, e)
                            stats["errors"] += 1
    except OSError:
        pass

    return stats


def run_gc(
    results_dir: Path,
    checkpoints_dir: Optional[Path],
    primary_metric: str,
    lake_db_path: Optional[Path] = None,
    keep_top: int = 50,
    keep_recent: int = 20,
    min_free_gb: float = 0,
    dry_run: bool = False,
    gc_results_enabled: bool = False,
) -> Dict[str, Any]:
    """Run garbage collection. Returns stats dict."""
    logger.info("=" * 50)
    logger.info("GARBAGE COLLECTION%s", " (DRY RUN)" if dry_run else "")
    logger.info("=" * 50)

    # Check if GC is needed (disk space gate)
    if min_free_gb > 0 and results_dir.exists():
        try:
            usage = shutil.disk_usage(results_dir)
            free_gb = usage.free / (1024 ** 3)
            if free_gb >= min_free_gb:
                logger.info("Disk has %.1fGB free (>= %.0fGB threshold), "
                            "skipping GC", free_gb, min_free_gb)
                return {"skipped": True, "free_gb": round(free_gb, 1)}
        except Exception:
            pass

    # Build the keep set
    keep_ids = get_top_idea_ids(results_dir, primary_metric,
                                lake_db_path, keep_top)
    logger.info("Top performers: %d ideas", len(keep_ids))

    recent_ids = get_recent_idea_ids(results_dir, keep_recent)
    keep_ids |= recent_ids
    logger.info("+ Recent completions: %d ideas", len(recent_ids))

    active_ids = get_active_idea_ids(results_dir)
    keep_ids |= active_ids
    if active_ids:
        logger.info("+ Currently active: %d ideas", len(active_ids))

    logger.info("Total protected: %d ideas", len(keep_ids))

    stats: Dict[str, Any] = {"checkpoints": {}, "results": {}}

    # GC checkpoints
    if checkpoints_dir:
        stats["checkpoints"] = gc_checkpoints(
            checkpoints_dir, keep_ids, dry_run=dry_run)
        cs = stats["checkpoints"]
        logger.info("Checkpoints: deleted=%d, kept=%d, errors=%d",
                     cs["deleted"], cs["kept"], cs["errors"])

    # GC results artifacts
    if gc_results_enabled:
        stats["results"] = gc_results(results_dir, keep_ids, dry_run=dry_run)
        rs = stats["results"]
        freed_mb = rs["freed_bytes"] / (1024 * 1024)
        logger.info("Results artifacts: deleted=%d, freed=%.1fMB, kept=%d",
                     rs["deleted_files"], freed_mb, rs["kept"])

    # Report final disk state
    if results_dir.exists():
        try:
            usage = shutil.disk_usage(results_dir)
            stats["free_gb"] = round(usage.free / (1024 ** 3), 1)
            logger.info("Disk free: %.1fGB", stats["free_gb"])
        except Exception:
            pass

    return stats


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="orze gc — garbage collect experiment checkpoints",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Dry run (show what would be deleted)
  python orze/orze_gc.py -c orze.yaml --dry-run

  # Delete non-top-50 checkpoints
  python orze/orze_gc.py -c orze.yaml --keep-top 50

  # In orze.yaml:
  gc:
    enabled: true
    checkpoints_dir: checkpoints
    keep_top: 50
    keep_recent: 20
    min_free_gb: 100
""",
    )

    parser.add_argument("-c", "--config", default="orze.yaml",
                        help="Path to orze.yaml")
    parser.add_argument("--checkpoints-dir", default="",
                        help="Checkpoints directory (overrides orze.yaml gc.checkpoints_dir)")
    parser.add_argument("--keep-top", type=int, default=0,
                        help="Keep top N by primary metric (overrides orze.yaml)")
    parser.add_argument("--keep-recent", type=int, default=0,
                        help="Also keep N most recently completed")
    parser.add_argument("--min-free-gb", type=float, default=0,
                        help="Only run if disk free < this (0 = always run)")
    parser.add_argument("--lake-db", default="",
                        help="Path to idea_lake.db")
    parser.add_argument("--gc-results", action="store_true",
                        help="Delete large artifacts (.pt) from results/ too")
    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be deleted without deleting")

    args = parser.parse_args()

    # Load orze.yaml
    cfg = {}
    config_path = Path(args.config)
    if config_path.exists():
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            logger.warning("Could not load %s: %s", args.config, e)

    gc_cfg = cfg.get("gc") or {}
    report_cfg = cfg.get("report") or {}

    results_dir = Path(cfg.get("results_dir", "results"))
    ideas_path = Path(cfg.get("ideas_file", "ideas.md"))
    primary_metric = report_cfg.get("primary_metric", "")

    checkpoints_dir = args.checkpoints_dir or gc_cfg.get("checkpoints_dir", "")
    if checkpoints_dir:
        checkpoints_dir = Path(checkpoints_dir)
    else:
        checkpoints_dir = None

    keep_top = args.keep_top or gc_cfg.get("keep_top", 50)
    keep_recent = args.keep_recent or gc_cfg.get("keep_recent", 20)
    min_free_gb = args.min_free_gb or gc_cfg.get("min_free_gb", 0)
    gc_results_enabled = args.gc_results or gc_cfg.get("results_artifacts", False)

    lake_db_path = Path(args.lake_db) if args.lake_db else \
        ideas_path.parent / "idea_lake.db"

    stats = run_gc(
        results_dir=results_dir,
        checkpoints_dir=checkpoints_dir,
        primary_metric=primary_metric,
        lake_db_path=lake_db_path,
        keep_top=keep_top,
        keep_recent=keep_recent,
        min_free_gb=min_free_gb,
        dry_run=args.dry_run,
        gc_results_enabled=gc_results_enabled,
    )

    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
