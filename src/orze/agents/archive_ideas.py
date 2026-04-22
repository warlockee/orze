#!/usr/bin/env python3
"""Archive completed/failed ideas from ideas.md into idea_lake.db.

Keeps only unclaimed (no results dir) ideas in ideas.md.
All archived ideas remain queryable via IdeaLake.

Usage:
    python orze/archive_ideas.py --ideas-md ideas.md --results-dir results --db idea_lake.db
    python orze/archive_ideas.py --ideas-md ideas.md --results-dir results --db idea_lake.db --dry-run
    python orze/archive_ideas.py --ideas-md ideas.md --results-dir results --db idea_lake.db --config orze.yaml
"""

import argparse
import json
import logging
import re
import shutil
from pathlib import Path

import yaml

from orze.core.fs import deep_get
from orze.idea_lake import IdeaLake

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("archive_ideas")

_IDEA_PATTERN = re.compile(r"^## (idea-[a-z0-9]+):\s*(.+?)$", re.MULTILINE)


def parse_all_ideas(text: str):
    """Parse ideas.md into list of (idea_id, title, raw_block, config_yaml)."""
    matches = list(_IDEA_PATTERN.finditer(text))
    ideas = []
    for i, m in enumerate(matches):
        idea_id = m.group(1)
        title = m.group(2).strip()
        block_start = m.start()
        block_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw_block = text[block_start:block_end].rstrip()

        # Extract priority
        pri_match = re.search(r"\*\*Priority\*\*:\s*(\w+)", raw_block)
        priority = pri_match.group(1).lower() if pri_match else "medium"

        # Extract category
        cat_match = re.search(r"\*\*Category\*\*:\s*(\w+)", raw_block)
        category = cat_match.group(1).lower() if cat_match else "architecture"

        # Extract parent
        par_match = re.search(r"\*\*Parent\*\*:\s*(idea-[a-z0-9]+|none)", raw_block)
        parent = par_match.group(1) if par_match and par_match.group(1) != "none" else None

        # Extract hypothesis
        hyp_match = re.search(r"\*\*Hypothesis\*\*:\s*(.+?)(?:\n-|\n\n|```)", raw_block, re.DOTALL)
        hypothesis = hyp_match.group(1).strip() if hyp_match else None

        # Extract YAML config
        yaml_match = re.search(r"```ya?ml\s*\n(.*?)```", raw_block, re.DOTALL)
        config_yaml = yaml_match.group(1) if yaml_match else ""

        ideas.append({
            "idea_id": idea_id,
            "title": title,
            "priority": priority,
            "category": category,
            "parent": parent,
            "hypothesis": hypothesis,
            "config_yaml": config_yaml,
            "raw_block": raw_block,
        })
    return ideas



def load_metrics(results_dir: Path, idea_id: str,
                 eval_reports: list, report_columns: list) -> dict:
    """Load metrics from result dir files for an idea.

    Args:
        eval_reports: list of eval report filenames to scan,
                      e.g. ["eval_report.json"]
        report_columns: list of column dicts from report config,
                        e.g. [{"key": "auc", "source": "report.json:metrics.auc_roc"}]
    """
    idea_dir = results_dir / idea_id
    metrics = {}

    # metrics.json — training status and time
    metrics_path = idea_dir / "metrics.json"
    if metrics_path.exists():
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            metrics["status"] = m.get("status", "COMPLETED").upper()
            tls = m.get("training_log_summary", {})
            if isinstance(tls, dict):
                metrics["training_time"] = tls.get("total_time")
            metrics["created_at"] = m.get("timestamp")
        except (json.JSONDecodeError, OSError):
            pass

    # Eval reports — extract metrics dynamically from config columns
    for report_file in eval_reports:
        report_path = idea_dir / report_file
        if not report_path.exists():
            continue
        try:
            r = json.loads(report_path.read_text(encoding="utf-8"))
            for col in report_columns:
                key = col.get("key", "")
                src = col.get("source", "")
                if ":" in src:
                    src_file, json_path = src.split(":", 1)
                    if src_file == report_file:
                        val = deep_get(r, json_path)
                        if val is not None:
                            metrics[key] = val
        except (json.JSONDecodeError, OSError):
            pass

    return metrics


def main():
    parser = argparse.ArgumentParser(description="Archive ideas to SQLite")
    parser.add_argument("--ideas-md", default="ideas.md", help="Path to ideas.md")
    parser.add_argument("--results-dir", default="orze_results", help="Path to results dir")
    parser.add_argument("--db", default=None, help="SQLite database path (default: from config or results_dir/idea_lake.db)")
    parser.add_argument("--config", default=None, help="Path to orze.yaml for report column config")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing")
    parser.add_argument("--keep-top", type=int, default=0,
                        help="Also keep top N ideas in ideas.md (0=only unclaimed)")
    args = parser.parse_args()

    ideas_path = Path(args.ideas_md)
    results_dir = Path(args.results_dir)

    # Load config for eval report names and column definitions
    eval_reports = []
    report_columns = []
    cfg = {}
    if args.config and Path(args.config).exists():
        try:
            cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
            eval_output = cfg.get("eval_output", "eval_report.json")
            eval_reports.append(eval_output)
            report_columns = cfg.get("report", {}).get("columns", [])
        except (yaml.YAMLError, OSError):
            pass
    if not eval_reports:
        eval_reports = ["eval_report.json"]

    # Resolve db path: CLI flag > config > results_dir/idea_lake.db
    if args.db:
        db_path = args.db
    elif cfg.get("idea_lake_db"):
        db_path = cfg["idea_lake_db"]
    else:
        db_path = str(results_dir / "idea_lake.db")

    logger.info("Reading %s...", ideas_path)
    text = ideas_path.read_text(encoding="utf-8")

    # Preserve header (everything before first ## idea-)
    first_idea = _IDEA_PATTERN.search(text)
    header = text[:first_idea.start()].rstrip() + "\n\n" if first_idea else ""

    ideas = parse_all_ideas(text)
    logger.info("Parsed %d ideas from ideas.md", len(ideas))

    # Classify: archive (has results dir) vs keep (unclaimed)
    to_archive = []
    to_keep = []
    for idea in ideas:
        idea_dir = results_dir / idea["idea_id"]
        if idea_dir.exists():
            metrics = load_metrics(results_dir, idea["idea_id"],
                                   eval_reports, report_columns)
            idea["metrics"] = metrics
            idea["status"] = metrics.get("status", "archived").lower()
            to_archive.append(idea)
        else:
            to_keep.append(idea)

    logger.info(
        "Archive: %d ideas (have results), Keep: %d ideas (unclaimed)",
        len(to_archive), len(to_keep),
    )

    # Stats
    statuses = {}
    for idea in to_archive:
        s = idea["status"]
        statuses[s] = statuses.get(s, 0) + 1
    for s, c in sorted(statuses.items()):
        logger.info("  %s: %d", s, c)

    with_eval = sum(1 for i in to_archive
                    if any(i["metrics"].get(col.get("key"))
                           is not None for col in report_columns))
    logger.info("  With eval metrics: %d", with_eval)

    if args.dry_run:
        logger.info("DRY RUN — no changes made")
        logger.info("Would archive %d ideas, keep %d in ideas.md", len(to_archive), len(to_keep))
        logger.info("ideas.md would shrink from %.1fMB to ~%.1fMB",
                     len(text) / 1e6,
                     sum(len(i["raw_block"]) for i in to_keep) / 1e6)
        return

    # Insert into SQLite
    logger.info("Inserting %d ideas into %s...", len(to_archive), db_path)
    lake = IdeaLake(db_path)

    # Bulk insert
    bulk_data = []
    for idea in to_archive:
        # Separate eval metrics from meta fields
        all_metrics = idea.get("metrics", {})
        eval_metrics = {k: v for k, v in all_metrics.items()
                        if k not in ("status", "training_time", "created_at")}
        bulk_data.append({
            "idea_id": idea["idea_id"],
            "title": idea["title"],
            "priority": idea["priority"],
            "category": idea["category"],
            "parent": idea["parent"],
            "hypothesis": idea["hypothesis"],
            "config_yaml": idea["config_yaml"],
            "raw_markdown": idea["raw_block"],
            "status": idea["status"],
            "eval_metrics": eval_metrics or None,
        })
    lake.bulk_insert(bulk_data)

    # Set next_id to max + 1
    max_id = 0
    for idea in ideas:
        try:
            num = int(idea["idea_id"].replace("idea-", ""))
            max_id = max(max_id, num)
        except ValueError:
            pass
    lake_max = lake.get_max_id_num()
    next_id = max(max_id, lake_max) + 1
    lake.set_next_id(next_id)
    logger.info("Set next_id sequence to %d", next_id)

    # Backup ideas.md
    backup_path = ideas_path.with_suffix(".md.archive_backup")
    shutil.copy2(ideas_path, backup_path)
    logger.info("Backed up ideas.md to %s", backup_path)

    # Rewrite ideas.md with only unclaimed ideas
    new_text = header
    for idea in to_keep:
        new_text += idea["raw_block"] + "\n\n"

    ideas_path.write_text(new_text, encoding="utf-8")
    logger.info(
        "Rewrote ideas.md: %d ideas (%.1fMB -> %.1fMB)",
        len(to_keep), len(text) / 1e6, len(new_text) / 1e6,
    )

    # Update archived index for report generation
    archived_index = results_dir / "_archived_index.json"
    existing_idx = {}
    if archived_index.exists():
        try:
            existing_idx = json.loads(archived_index.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    for idea in to_archive:
        if idea.get("status") == "completed":
            existing_idx[idea["idea_id"]] = idea["title"]
    archived_index.write_text(json.dumps(existing_idx), encoding="utf-8")
    logger.info("Updated archived index: %d entries", len(existing_idx))

    logger.info("Done! Lake has %d ideas total", lake.count())
    lake.close()


if __name__ == "__main__":
    main()
