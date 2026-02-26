import sys
import os
from pathlib import Path
import yaml
import json

# Setup paths
_SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_SCRIPT_DIR))

from farm import notify, update_report, parse_ideas, deep_get
from idea_lake import IdeaLake

def main():
    with open("orze.yaml") as f:
        cfg = yaml.safe_load(f)
    
    results_dir = Path(cfg["results_dir"])
    lake = IdeaLake("idea_lake.db")
    ideas = parse_ideas(cfg["ideas_file"])
    
    print("Generating report...")
    completed_rows = update_report(results_dir, ideas, cfg, lake=lake)
    
    # Calculate counts
    counts = {"COMPLETED": 0, "FAILED": 0, "QUEUED": 0, "IN_PROGRESS": 0}
    for r in completed_rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
    
    # Machine status (mock)
    machines = [{"host": "manual-trigger", "gpus_busy": 0, "gpus_total": 8, "utilization": 0}]
    
    # Calculate counts from DB
    counts = {"COMPLETED": 0, "FAILED": 0, "QUEUED": 0, "IN_PROGRESS": 0}
    all_db_ids = lake.get_all_ids()
    for aid in all_db_ids:
        item = lake.get(aid)
        st = item.get("status", "archived").upper()
        if st in counts:
            counts[st] += 1
        elif st == "ARCHIVED":
            counts["COMPLETED"] += 1
            
    # Active count from heartbeats
    active_count = 0
    try:
        with open(results_dir / "_admin_cache.json") as f:
            admin_data = json.load(f)
            for hb in admin_data.get("fleet", {}).get("heartbeats", []):
                if hb.get("status") == "online":
                    active_count += len(hb.get("active", []))
    except (json.JSONDecodeError, OSError):
        pass

    # Machine status from heartbeats
    machines = []
    try:
        for hb in admin_data.get("fleet", {}).get("heartbeats", []):
            machines.append({
                "host": hb["host"],
                "gpus_busy": len(hb.get("active", [])),
                "gpus_total": len(hb.get("gpu_info", [])),
                "utilization": hb.get("gpu_info", [{}])[0].get("utilization_pct", 0)
            })
    except Exception:
        machines = [{"host": "fleet-total", "gpus_busy": active_count, "gpus_total": 16, "utilization": "?"}]
    
    # Build leaderboard for notification
    primary = cfg["report"].get("primary_metric", "score")
    leaderboard = []
    for r in completed_rows:
        if r["status"] == "COMPLETED":
            # Extract the actual value for the leaderboard
            val = r.get("primary_val")
            # SYSTEMATIC FIX: No fallbacks.
            
            leaderboard.append({
                "id": r["id"],
                "title": r["title"],
                "value": val
            })
    
    # Sort leaderboard by value desc (mirroring farm.py systematic tie-breaker)
    leaderboard.sort(key=lambda x: x["value"] if x["value"] is not None else -1, reverse=True)
    
    print(f"Sending manual report to Telegram (total={len(all_db_ids)}, active={active_count})...")
    notify("report", {
        "title": "Fleet Status Update",
        "completed": counts.get("COMPLETED", 0),
        "failed": counts.get("FAILED", 0),
        "active_count": active_count,
        "queued": counts.get("QUEUED", 0),
        "metric_name": primary,
        "leaderboard": leaderboard[:10],
        "machines": machines
    }, cfg)
    
    lake.close()
    print("Done!")

if __name__ == "__main__":
    main()
