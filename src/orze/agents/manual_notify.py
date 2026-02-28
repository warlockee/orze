"""Manually trigger a notification report.

Usage:
    python -m orze.agents.manual_notify -c orze.yaml
"""

import argparse
import json
from pathlib import Path

import yaml

from orze.core.ideas import parse_ideas
from orze.idea_lake import IdeaLake
from orze.reporting.leaderboard import update_report
from orze.reporting.notifications import notify


def main():
    parser = argparse.ArgumentParser(description="Send a manual status report")
    parser.add_argument("-c", "--config", default="orze.yaml", help="Path to orze.yaml")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    results_dir = Path(cfg["results_dir"])
    ideas_file = cfg.get("ideas_file", "ideas.md")
    lake_path = Path(ideas_file).parent / "idea_lake.db"
    lake = IdeaLake(str(lake_path)) if lake_path.exists() else None
    ideas = parse_ideas(ideas_file)

    print("Generating report...")
    completed_rows = update_report(results_dir, ideas, cfg, lake=lake)

    # Calculate counts from DB if available
    counts = {"COMPLETED": 0, "FAILED": 0, "QUEUED": 0, "IN_PROGRESS": 0}
    if lake:
        all_db_ids = lake.get_all_ids()
        for aid in all_db_ids:
            item = lake.get(aid)
            st = (item.get("status", "archived") or "archived").upper()
            if st in counts:
                counts[st] += 1
            elif st == "ARCHIVED":
                counts["COMPLETED"] += 1
    else:
        for r in completed_rows:
            st = r.get("status", "UNKNOWN")
            counts[st] = counts.get(st, 0) + 1

    # Active count and machine status from heartbeats
    active_count = 0
    admin_data = {}
    try:
        with open(results_dir / "_admin_cache.json", encoding="utf-8") as f:
            admin_data = json.load(f)
            for hb in admin_data.get("nodes", {}).get("heartbeats", []):
                if hb.get("status") == "online":
                    active_count += len(hb.get("active", []))
    except (json.JSONDecodeError, OSError):
        pass

    machines = []
    try:
        for hb in admin_data.get("nodes", {}).get("heartbeats", []):
            machines.append({
                "host": hb.get("host", "?"),
                "gpus_busy": len(hb.get("active", [])),
                "gpus_total": len(hb.get("gpu_info", [])),
                "utilization": (hb.get("gpu_info") or [{}])[0].get("utilization_pct", 0),
            })
    except Exception:
        if active_count:
            machines = [{"host": "unknown", "gpus_busy": active_count,
                         "gpus_total": active_count, "utilization": "?"}]

    # Build leaderboard
    report_cfg = cfg.get("report", {})
    primary = report_cfg.get("primary_metric", "score")
    leaderboard = []
    for r in completed_rows:
        if r.get("status") == "COMPLETED":
            leaderboard.append({
                "id": r["id"],
                "title": r["title"],
                "value": r.get("primary_val"),
            })

    leaderboard.sort(
        key=lambda x: x["value"] if x["value"] is not None else -1,
        reverse=True,
    )

    total = len(lake.get_all_ids()) if lake else len(completed_rows)
    print(f"Sending report (total={total}, active={active_count})...")
    notify("report", {
        "title": "Status Update",
        "completed": counts.get("COMPLETED", 0),
        "failed": counts.get("FAILED", 0),
        "active_count": active_count,
        "queued": counts.get("QUEUED", 0),
        "metric_name": primary,
        "leaderboard": leaderboard[:10],
        "machines": machines,
    }, cfg)

    if lake:
        lake.close()
    print("Done!")


if __name__ == "__main__":
    main()
