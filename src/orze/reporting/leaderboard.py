import os
import json
import logging
import re
import time
import socket
import shutil
import datetime
from typing import Dict, Optional
from pathlib import Path
from orze.core.fs import deep_get, atomic_write
from orze.core.ideas import expand_sweeps
from orze.core.config import DEFAULT_CONFIG
from orze.reporting.state import _read_all_heartbeats

try:
    from orze.idea_lake import IdeaLake
except ImportError:
    IdeaLake = None

logger = logging.getLogger("orze")


def _resolve_primary_metric(cfg: dict, eval_file: str, eval_data: dict):
    """Resolve the primary metric value from eval data using report config."""
    primary = cfg.get("report", {}).get("primary_metric", "test_accuracy")
    columns = cfg.get("report", {}).get("columns", [])
    for col in columns:
        if col.get("key") == primary:
            src = col.get("source", "")
            if ":" in src:
                src_file, json_path = src.split(":", 1)
                if src_file == eval_file:
                    return deep_get(eval_data, json_path)
    # Fallback: try metrics.{primary} directly
    return deep_get(eval_data, f"metrics.{primary}")


def _read_metric_value(results_dir: Path, idea_id: str, col: dict):
    """Read a metric value for a column. Supports 'source' field."""
    key = col.get("key")
    source = col.get("source", "")
    if not key:
        return None

    if source and ":" in source:
        filename, dotpath = source.split(":", 1)
        filepath = results_dir / idea_id / filename
        if filepath.exists():
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                return deep_get(data, dotpath)
            except (json.JSONDecodeError, KeyError, OSError, UnicodeDecodeError):
                return None
        return None

    metrics_path = results_dir / idea_id / "metrics.json"
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            return deep_get(data, key) if "." in key else data.get(key)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None
    return None


def update_report(results_dir: Path, ideas: Dict[str, dict],
                  cfg: dict, lake: Optional[IdeaLake] = None) -> list:
    """Generate a configurable leaderboard report.md from all results.
    Returns sorted list of completed row dicts."""
    report_cfg = cfg.get("report") or DEFAULT_CONFIG["report"]
    primary_metric = report_cfg.get("primary_metric") or "test_accuracy"
    sort_order = report_cfg.get("sort") or "descending"
    columns = report_cfg.get("columns") or DEFAULT_CONFIG["report"]["columns"]
    title = report_cfg.get("title") or "Orze Report"
    reverse = sort_order == "descending"
    _sentinel = float("-inf") if reverse else float("inf")

    def _safe_float(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return _sentinel

    # Tie-breaker: use secondary_metric from config, else just primary
    secondary_metric = report_cfg.get("secondary_metric")

    def _get_tiebreaker_sort_key(r):
        pv = _safe_float(r.get("primary_val"))
        if secondary_metric:
            sv = _safe_float(
                r.get("values", {}).get(secondary_metric)
                or deep_get(r.get("metrics", {}), secondary_metric)
                or 0.0)
        else:
            sv = _safe_float(0.0)
        return (pv, sv)

    # --- Load results cache ---
    cache_path = results_dir / "_results_cache.json"
    cache = {}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass

    rows = []
    def _id_sort_key(x):
        return x

    # Include archived ideas from cached index
    all_ideas = dict(ideas)
    if lake:
        # DB is source of truth for full idea set (archived + hot)
        db_ids = lake.get_all_ids()
        # In memory ideas (hot) take precedence for config, but DB knows about everyone
        for aid in db_ids:
            if aid not in all_ideas:
                # Stub for report walker — title will be fetched from DB below
                all_ideas[aid] = {"title": "...", "priority": "archived"}
    else:
        archived_index = results_dir / "_archived_index.json"
        if archived_index.exists():
            try:
                idx = json.loads(archived_index.read_text(encoding="utf-8"))
                for arch_id, arch_title in idx.items():
                    if arch_id not in all_ideas:
                        all_ideas[arch_id] = {"title": arch_title, "priority": "archived"}
            except (json.JSONDecodeError, OSError):
                pass

    # Determine report title
    report_title = report_cfg.get("title") or "Orze Report"

    updated_cache = False
    for idea_id in sorted(all_ideas.keys(), key=_id_sort_key):
        idea_dir = results_dir / idea_id
        
        # Determine base title/data (from memory or DB)
        idea_info = all_ideas.get(idea_id, {"title": idea_id})
        curr_idea_title = idea_info["title"]
        
        if lake and curr_idea_title == "...":
            # Lazy fetch title from DB for archived ideas
            db_idea = lake.get(idea_id)
            if db_idea:
                curr_idea_title = db_idea["title"]
                all_ideas[idea_id]["title"] = curr_idea_title

        if not idea_dir.exists():
            rows.append({"id": idea_id, "title": curr_idea_title,
                         "status": "QUEUED", "values": {}})
            continue

        metrics_path = idea_dir / "metrics.json"
        if not metrics_path.exists():
            rows.append({"id": idea_id, "title": curr_idea_title,
                         "status": "IN_PROGRESS", "values": {}})
            continue

        # Check cache: invalidate if metrics.json OR any source file changed
        mtime = metrics_path.stat().st_mtime
        # Also track mtime of external source files (e.g. fedex_test_report.json)
        for col in columns:
            src = col.get("source", "")
            if src and ":" in src:
                src_file = idea_dir / src.split(":", 1)[0]
                if src_file.exists():
                    mtime = max(mtime, src_file.stat().st_mtime)
        cached = cache.get(idea_id)
        if cached and cached.get("mtime") == mtime:
            rows.append(cached["row"])
            continue

        # Cache miss or stale: read and parse
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            metrics = {"status": "FAILED", "error": "corrupt metrics.json"}

        values = {}
        for col in columns:
            key = col.get("key")
            if not key:
                continue
            values[key] = _read_metric_value(results_dir, idea_id, col)

        primary_val = values.get(primary_metric)
        # SYSTEMATIC FIX: No deceptive fallbacks. Only use verified report values.
        # If the external report is missing or zero, we show 0.0 or None.
        
        row_data = {
            "id": idea_id, "title": curr_idea_title,
            "status": metrics.get("status", "UNKNOWN"),
            "values": values,
            "primary_val": primary_val,
            "metrics": metrics,
        }
        rows.append(row_data)
        cache[idea_id] = {"mtime": mtime, "row": row_data}
        updated_cache = True

    if updated_cache:
        try:
            atomic_write(cache_path, json.dumps(cache))
        except OSError:
            pass

    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# {report_title}",
        f"**Updated:** {now} | **Host:** {socket.gethostname()}",
        "",
        "## Pipeline Status",
        "| Total | Completed | Failed | In Progress | Queued |",
        "|-------|-----------|--------|-------------|--------|",
        f"| {len(rows)} | {counts.get('COMPLETED', 0)} "
        f"| {counts.get('FAILED', 0)} "
        f"| {counts.get('IN_PROGRESS', 0)} | {counts.get('QUEUED', 0)} |",
        "",
        "## Results",
        "",
    ]

    completed = [r for r in rows if r["status"] == "COMPLETED"]
    completed.sort(key=_get_tiebreaker_sort_key, reverse=reverse)

    # --- Sweep grouping: split into standalone + sweep groups ---
    standalone = []
    sweep_groups = {}  # parent_id -> list of sub-run rows
    for r in completed:
        parent_id = None
        if "-ht-" in r["id"]:
            parent_id = r["id"].split("-ht-", 1)[0]
        elif "~" in r["id"]:
            parent_id = r["id"].split("~", 1)[0]
            
        if parent_id:
            sweep_groups.setdefault(parent_id, []).append(r)
        else:
            standalone.append(r)

    # Build main table: standalone + best of each sweep group
    main_rows = list(standalone)
    for parent_id, children in sweep_groups.items():
        children.sort(key=_get_tiebreaker_sort_key,
                      reverse=reverse)
        best = dict(children[0])
        best["title"] = f"{best['title']} (best of {len(children)})"
        main_rows.append(best)

    main_rows.sort(key=_get_tiebreaker_sort_key,
                   reverse=reverse)

    if main_rows:
        header = "| Rank | Idea | Title"
        sep = "|------|------|------"
        for col in columns:
            label = col.get("label", col.get("key", "?"))
            header += f" | {label}"
            sep += " |" + "-" * max(6, len(str(label)))
        header += " |"
        sep += " |"
        lines.append(header)
        lines.append(sep)

        for rank, r in enumerate(main_rows, 1):
            row = f"| {rank} | {r['id']} | {r['title'][:50]}"
            for col in columns:
                key = col.get("key")
                if not key:
                    row += " | —"
                    continue
                val = r["values"].get(key)
                if val is not None:
                    fmt = col.get("fmt", "")
                    try:
                        row += f" | {val:{fmt}}"
                    except (ValueError, TypeError):
                        row += f" | {val}"
                else:
                    row += " | —"
            row += " |"
            lines.append(row)
        lines.append("")

    # --- Sweep Details section ---
    if sweep_groups:
        lines.append("## Sweep Details")
        lines.append("")
        for parent_id in sorted(sweep_groups.keys(), key=_id_sort_key):
            children = sweep_groups[parent_id]
            children.sort(key=lambda r: _safe_float(r.get("primary_val")),
                          reverse=reverse)
            lines.append(f"### {parent_id} ({len(children)} variants)")
            lines.append(f"| Rank | Sub-run | {primary_metric} |")
            lines.append("|------|---------|" + "-" * max(6, len(primary_metric)) + "|")
            for i, r in enumerate(children, 1):
                pv = r.get("primary_val", "—")
                if isinstance(pv, float):
                    pv = f"{pv:.4f}"
                
                # Extract suffix for display
                sub_id = r["id"]
                if "-ht-" in sub_id:
                    suffix = "ht-" + sub_id.split("-ht-", 1)[1]
                elif "~" in sub_id:
                    suffix = sub_id.split("~", 1)[1]
                else:
                    suffix = sub_id
                    
                lines.append(f"| {i} | {suffix} | {pv} |")
            lines.append("")

    failed = [r for r in rows if r["status"] == "FAILED"]
    if failed:
        lines.append("## Failed")
        for r in failed:
            err = r.get("metrics", {}).get("error", "unknown")
            lines.append(
                f"- **{r['id']}**: {r['title'][:50]} — {str(err)[:80]}")
        lines.append("")

    queued = [r for r in rows if r["status"] == "QUEUED"]
    if queued:
        lines.append(f"## Queue ({len(queued)} ideas)")
        for r in queued[:20]:
            pri = all_ideas.get(r["id"], {}).get("priority", "medium")
            lines.append(f"- **{r['id']}** [{pri}]: {r['title'][:60]}")
        if len(queued) > 20:
            lines.append(f"- ... and {len(queued) - 20} more")
        lines.append("")

    report_path = results_dir / "report.md"
    atomic_write(report_path, "\n".join(lines))

    # Write leaderboard cache for admin panel (avoids expensive rescan)
    lb_entries = []
    for r in main_rows[:20]:
        lb_entries.append({
            "idea_id": r["id"],
            "title": r["title"],
            "metric_value": r.get("primary_val"),
            "training_time": r["values"].get("training_time"),
            "status": "COMPLETED",
            "eval_metrics": r["values"],
        })
    lb_path = results_dir / "_leaderboard.json"
    atomic_write(lb_path, json.dumps({"top": lb_entries, "metric": primary_metric},
                                     default=str))

    logger.info("Report updated: %d completed, %d queued, %d failed",
                counts.get("COMPLETED", 0), counts.get("QUEUED", 0),
                counts.get("FAILED", 0))

    return completed


def write_admin_cache(results_dir: Path, ideas: dict, cfg: dict):
    """Write pre-aggregated _admin_cache.json for instant admin panel access.

    Aggregates nodes (heartbeats + GPU info), queue (with status),
    and alerts — so the admin server never needs to scan the filesystem.
    """
    now = time.time()

    # --- Nodes: enrich heartbeats ---
    raw_hb = _read_all_heartbeats(results_dir, stale_seconds=600)
    heartbeats = []
    for hb in raw_hb:
        age = now - hb.get("epoch", 0)
        status = "online"
        if age > 300:
            status = "offline"
        elif age > 120:
            status = "degraded"
        heartbeats.append({
            **hb,
            "status": status,
            "heartbeat_age_sec": round(age, 1),
        })

    # --- Queue: expanded ideas with status ---
    sweep_max = cfg.get("sweep", {}).get("max_combos", 20)
    expanded = expand_sweeps(dict(ideas), max_combos=sweep_max)
    all_statuses: dict = {}
    queue_items = []
    for idea_id, idea in expanded.items():
        idea_dir = results_dir / idea_id
        idea_status = "pending"
        if idea_dir.exists():
            mpath = idea_dir / "metrics.json"
            if mpath.exists():
                try:
                    m = json.loads(mpath.read_text(encoding="utf-8"))
                    idea_status = m.get("status", "COMPLETED").lower()
                except (json.JSONDecodeError, OSError):
                    idea_status = "running"
            else:
                idea_status = "running"
        all_statuses[idea_status] = all_statuses.get(idea_status, 0) + 1
        queue_items.append({
            "idea_id": idea_id,
            "title": idea.get("title", ""),
            "priority": idea.get("priority", "medium"),
            "status": idea_status,
            "config": idea.get("config", {}),
            "sweep_parent": idea.get("_sweep_parent"),
        })

    # --- Alerts ---
    alerts = []
    two_hours_ago = now - 7200
    try:
        with os.scandir(results_dir) as it:
            for entry in it:
                if not entry.is_dir() or not entry.name.startswith("idea-"):
                    continue
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    continue
                if mtime < two_hours_ago:
                    continue
                mpath = Path(entry.path) / "metrics.json"
                if not mpath.exists():
                    continue
                try:
                    m = json.loads(mpath.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                if m.get("status") not in ("FAILED", "ERROR"):
                    continue
                alerts.append({
                    "type": "failure",
                    "idea_id": entry.name,
                    "error": str(m.get("error", m.get("status", "")))[:200],
                    "minutes_ago": round((now - mtime) / 60, 1),
                })
    except OSError:
        pass

    for hb in heartbeats:
        if hb.get("status") == "offline":
            alerts.append({
                "type": "stale_host",
                "host": hb.get("host", "unknown"),
                "minutes_ago": round(hb.get("heartbeat_age_sec", 0) / 60, 1),
            })

    try:
        usage = shutil.disk_usage(results_dir)
        disk_free = round(usage.free / (1024 ** 3), 1)
        if disk_free < 50:
            alerts.append({"type": "low_disk", "disk_free_gb": disk_free})
    except Exception:
        pass

    cache = {
        "nodes": {"heartbeats": heartbeats, "local_gpus": []},
        "queue": {"items": queue_items, "counts": all_statuses,
                  "total_all": sum(all_statuses.values())},
        "alerts": {"alerts": alerts, "count": len(alerts)},
        "epoch": now,
    }
    atomic_write(results_dir / "_admin_cache.json",
                 json.dumps(cache, default=str))


def _format_report_text(data: dict) -> str:
    """Format a periodic report summary.

    data keys:
      title, completed, failed, active_count, queued,
      leaderboard: [{id, title, value}],
      metric_name,
      machines: [{host, gpus_busy, gpus_total, utilization}]
    """
    c = data.get("completed", 0)
    f = data.get("failed", 0)
    a = data.get("active_count", 0)
    q = data.get("queued", 0)
    title = data.get("title", "Report")
    metric = data.get("metric_name", "score")
    board = data.get("leaderboard", [])
    machines = data.get("machines", [])

    lines = [title]
    lines.append(f"{c} completed | {f} failed | {a} active | {q} queued")
    lines.append("")

    # Machine status
    if machines:
        lines.append("Machines:")
        for m in machines:
            host = m.get("host", "?")
            busy = m.get("gpus_busy", 0)
            total = m.get("gpus_total", 0)
            util = m.get("utilization", "?")
            lines.append(f"  {host}: {busy}/{total} GPUs, {util}% util")
        lines.append("")

    # Top 10 leaderboard
    if board:
        lines.append(f"Top {len(board)} ({metric}):")
        for i, entry in enumerate(board, 1):
            val = entry.get("value")
            val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
            eid = entry.get("id", "?")
            title_short = entry.get("title", "")[:25]
            lines.append(f"  #{i} {eid}: {val_str} {title_short}")

    return "\n".join(lines)
