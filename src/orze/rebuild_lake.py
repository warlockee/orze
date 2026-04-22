"""Rebuild idea_lake.db from existing results directories.

Usage:
    python -m orze.agents.rebuild_lake
    python -m orze.agents.rebuild_lake --results-dir results --db idea_lake.db
"""

import argparse
import json
import logging
from pathlib import Path

import yaml

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger("rebuild_lake")


def rebuild(results_dir: Path, db_path: Path):
    from orze.idea_lake import IdeaLake
    lake = IdeaLake(str(db_path))

    idea_dirs = sorted([d for d in results_dir.iterdir()
                        if d.is_dir() and d.name.startswith("idea-")])
    logger.info("Found %d idea directories in %s", len(idea_dirs), results_dir)

    count = 0
    for d in idea_dirs:
        idea_id = d.name
        metrics_path = d / "metrics.json"
        config_path = d / "resolved_config.yaml"

        if not metrics_path.exists():
            continue

        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            config_yaml = ""
            if config_path.exists():
                config_yaml = config_path.read_text(encoding="utf-8")

            title = metrics.get("idea_title") or idea_id
            status = metrics.get("status", "archived").lower()

            # Extract eval metrics from any *_report.json files
            eval_metrics = {}
            for report_path in sorted(d.glob("*_report.json")):
                try:
                    rd = json.loads(report_path.read_text(encoding="utf-8"))
                    rm = rd.get("metrics", {})
                    prefix = report_path.stem.replace("_report", "")
                    for k, v in rm.items():
                        eval_metrics[f"{prefix}_{k}"] = v
                except (json.JSONDecodeError, OSError):
                    pass

            # Fallback: pull from metrics.json test_metrics
            if not eval_metrics and "test_metrics" in metrics:
                for k, v in metrics["test_metrics"].items():
                    eval_metrics[f"test_{k}"] = v

            # Infer approach family from config (v4.0: simple on-the-fly)
            approach_family = "other"
            if config_yaml:
                try:
                    cfg_obj = yaml.safe_load(config_yaml)
                    if isinstance(cfg_obj, dict):
                        from orze.research.context_builder import (
                            _flatten, _infer_family,
                        )
                        approach_family = _infer_family(_flatten(cfg_obj))
                except Exception:
                    pass

            lake.insert(
                idea_id=idea_id,
                title=title,
                config_yaml=config_yaml,
                raw_markdown="",
                eval_metrics=eval_metrics,
                status=status,
                created_at=metrics.get("timestamp"),
                approach_family=approach_family,
            )
            count += 1
            if count % 1000 == 0:
                logger.info("Processed %d ideas...", count)
        except Exception as e:
            logger.warning("Failed to process %s: %s", idea_id, e)

    lake.close()
    logger.info("Rebuild complete. Total %d ideas indexed.", count)


def main():
    parser = argparse.ArgumentParser(description="Rebuild idea_lake.db from results")
    parser.add_argument("--results-dir", default="orze_results", help="Path to results dir")
    parser.add_argument("--db", default=None, help="SQLite database path (default: from config or results_dir/idea_lake.db)")
    parser.add_argument("-c", "--config", default=None, help="Path to orze.yaml")
    args = parser.parse_args()

    # Resolve db path: CLI flag > config > results_dir/idea_lake.db
    results_dir = Path(args.results_dir)
    if args.db:
        db_path = Path(args.db)
    elif args.config and Path(args.config).exists():
        try:
            cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
            db_path = Path(cfg.get("idea_lake_db") or (results_dir / "idea_lake.db"))
            results_dir = Path(cfg.get("results_dir", "orze_results"))
        except Exception:
            db_path = results_dir / "idea_lake.db"
    else:
        db_path = results_dir / "idea_lake.db"

    rebuild(results_dir, db_path)


if __name__ == "__main__":
    main()
