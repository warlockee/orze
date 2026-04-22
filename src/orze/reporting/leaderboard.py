"""Leaderboard report generation from experiment results.

CALLING SPEC:
    update_report(results_dir: Path, ideas: Dict[str, dict], cfg: dict,
                  lake: Optional[IdeaLake] = None) -> list
        Generate report.md leaderboard and JSON caches from all results.
        Returns sorted list of completed row dicts. Reads metrics.json per
        idea, caches results, handles sweep grouping and filtered views.
        cfg must contain 'report' key with primary_metric, columns, etc.

    write_admin_cache(results_dir: Path, ideas: dict, cfg: dict) -> None
        Write _admin_cache.json with pre-aggregated nodes, queue, and alerts
        for the admin panel. Reads heartbeats, expands sweeps, scans for
        recent failures.

    _resolve_primary_metric(cfg: dict, eval_file: str, eval_data: dict) -> Any
        Extract the primary metric value from eval_data using report column
        source mappings. Falls back to metrics.<primary_metric> dotpath.

    _format_report_text(data: dict) -> str
        Format a periodic report summary as plain text. data keys: title,
        completed, failed, active_count, queued, leaderboard (list of
        {id, title, value}), metric_name, machines (list of {host,
        gpus_busy, gpus_total, utilization}).
"""
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
from orze.core.config import DEFAULT_CONFIG, orze_path
from orze.reporting.state import _read_all_heartbeats

try:
    from orze.idea_lake import IdeaLake
except ImportError:
    IdeaLake = None

logger = logging.getLogger("orze")


_CMP_OPS = {
    "$lt": lambda a, b: a < b,
    "$lte": lambda a, b: a <= b,
    "$gt": lambda a, b: a > b,
    "$gte": lambda a, b: a >= b,
}


def _matches_view_filter(results_dir: Path, idea_id: str, view_filter: dict) -> bool:
    """Check if an idea's resolved_config matches all view filter conditions.

    Filter keys are dotpaths into resolved_config.yaml.
    Values can be:
      - a single value: exact match
      - a list: any-of match
      - a dict with comparison operators: {"$lte": 100}
    """
    config_path = results_dir / idea_id / "resolved_config.yaml"
    if not config_path.exists():
        return False
    try:
        import yaml
        config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if not isinstance(config, dict):
        return False

    for dotpath, expected in view_filter.items():
        actual = deep_get(config, dotpath)
        if isinstance(expected, dict):
            for op, threshold in expected.items():
                cmp_fn = _CMP_OPS.get(op)
                if cmp_fn is None:
                    return False
                if actual is None:
                    return False
                try:
                    if not cmp_fn(float(actual), float(threshold)):
                        return False
                except (ValueError, TypeError):
                    return False
        elif isinstance(expected, list):
            if actual not in expected:
                return False
        else:
            if actual != expected:
                return False
    return True


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


def _analyze_config_diversity(results_dir: Path, completed_ids: list, max_dims: int = 15) -> str:
    """Analyze resolved config diversity. Returns markdown table string."""
    try:
        import yaml as _yaml
    except ImportError:
        return ""

    # Collect values for each key path across all completed ideas
    key_values = {}  # dotpath -> list of values

    def _walk(d, prefix="", depth=0):
        if depth > 3 or not isinstance(d, dict):
            return
        for k, v in d.items():
            if k.startswith("_") or "/" in k:
                continue
            dotpath = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
            if isinstance(v, dict):
                _walk(v, dotpath, depth + 1)
            elif isinstance(v, (str, int, float, bool, type(None))):
                key_values.setdefault(dotpath, []).append(v)
            # skip lists and other non-scalar types

    for idea_id in completed_ids:
        cfg_path = results_dir / idea_id / "resolved_config.yaml"
        if not cfg_path.exists():
            continue
        try:
            config = _yaml.safe_load(cfg_path.read_text(encoding="utf-8"))
            if isinstance(config, dict):
                _walk(config)
        except Exception:
            continue

    if not key_values:
        return ""

    # Analyze each dimension
    dim_stats = []
    for dotpath, vals in key_values.items():
        n_unique = len(set(str(v) for v in vals))
        # Find most common value
        counts = {}
        for v in vals:
            sv = str(v)
            counts[sv] = counts.get(sv, 0) + 1
        dominant_val = max(counts, key=counts.get)
        dominant_pct = counts[dominant_val] / len(vals) * 100

        # Collect all unique values for display
        all_unique = sorted(set(str(v) for v in vals))

        # Filter: one value dominates >70% OR n_unique < 5
        if dominant_pct > 70 or n_unique < 5:
            dim_stats.append({
                "dim": dotpath,
                "n_unique": n_unique,
                "dominant": dominant_val,
                "dominant_pct": dominant_pct,
                "all_values": all_unique,
            })

    if not dim_stats:
        return ""

    # Sort by n_unique ascending (least diverse first)
    dim_stats.sort(key=lambda x: x["n_unique"])
    dim_stats = dim_stats[:max_dims]

    lines = [
        "## Config Diversity",
        "",
        "| Dimension | Unique | Dominant (%) | All Values |",
        "|-----------|--------|--------------|------------|",
    ]
    for s in dim_stats:
        all_vals_str = ", ".join(s["all_values"][:10])
        if len(s["all_values"]) > 10:
            all_vals_str += f", ... (+{len(s['all_values']) - 10})"
        lines.append(
            f"| {s['dim']} | {s['n_unique']} "
            f"| {s['dominant']} ({s['dominant_pct']:.0f}%) "
            f"| {all_vals_str} |"
        )
    lines.append("")
    return "\n".join(lines)


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

    # Filter out experiments with too few per-dataset metrics
    min_ds = report_cfg.get("min_datasets", 0)
    if min_ds > 0:
        filtered = []
        for r in rows:
            if r["status"] != "COMPLETED":
                filtered.append(r)
                continue
            ds_count = sum(1 for k, v in (r.get("metrics") or {}).items()
                          if k.startswith("wer_") and v is not None
                          and isinstance(v, (int, float)))
            if ds_count >= min_ds:
                filtered.append(r)
        rows = filtered

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

    # Exclude experiments that completed trivially without producing
    # meaningful results (e.g. all eval datasets missing, script exited
    # immediately). Without this filter, a 0-valued primary metric from
    # a vacuous run can top an ascending-sorted leaderboard.
    def _has_real_result(r):
        m = r.get("metrics", {})
        pv = _safe_float(r.get("primary_val"))
        t = m.get("training_time", 999)
        # Skip if primary metric is exactly 0 and run finished in < 30s
        if pv == 0.0 and t < 30:
            return False
        return True

    completed = [r for r in completed if _has_real_result(r)]
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

    # --- Score Ceiling Detection ---
    ceiling_k = report_cfg.get("ceiling_k", 20)
    ceiling_threshold = report_cfg.get("ceiling_std_threshold", 0.015)
    ceiling_min = report_cfg.get("ceiling_min_ideas", 30)
    if len(completed) >= ceiling_min:
        primary_vals = []
        for r in completed:
            try:
                primary_vals.append(float(r.get("primary_val")))
            except (ValueError, TypeError):
                pass
        if len(primary_vals) >= ceiling_k:
            top_k = primary_vals[:ceiling_k]  # already sorted
            top_std = (sum((v - sum(top_k) / len(top_k)) ** 2 for v in top_k)
                       / len(top_k)) ** 0.5
            top_min = min(top_k)
            top_max = max(top_k)
            if top_std < ceiling_threshold:
                lines.append("## Score Ceiling Warning")
                lines.append(
                    f"Top {ceiling_k} scores cluster tightly "
                    f"(std={top_std:.4f}, range={top_min:.4f}-{top_max:.4f}).")
                lines.append(
                    "Current approach may have reached a fundamental ceiling.")
                lines.append(
                    "Consider a fundamentally different architecture or data strategy.")
                lines.append("")

    # --- Config Diversity ---
    if len(completed) >= 10:
        completed_ids = [r["id"] for r in completed]
        diversity_md = _analyze_config_diversity(results_dir, completed_ids)
        if diversity_md:
            lines.append(diversity_md)

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

    # --- Filtered view leaderboards ---
    views = report_cfg.get("views") or []
    view_names = []
    for view in views:
        vname = view.get("name")
        vtitle = view.get("title", vname)
        vfilter = view.get("filter", {})
        if not vname or not vfilter:
            continue
        view_names.append(vname)

        filtered_rows = [
            r for r in main_rows
            if _matches_view_filter(results_dir, r["id"], vfilter)
        ]

        # Write filtered leaderboard JSON
        view_entries = []
        for r in filtered_rows[:20]:
            view_entries.append({
                "idea_id": r["id"],
                "title": r["title"],
                "metric_value": r.get("primary_val"),
                "training_time": r["values"].get("training_time"),
                "status": "COMPLETED",
                "eval_metrics": r["values"],
            })
        view_lb_path = results_dir / f"_leaderboard_{vname}.json"
        atomic_write(view_lb_path, json.dumps(
            {"top": view_entries, "metric": primary_metric, "view": vname,
             "title": vtitle},
            default=str))

        # Append view section to report.md
        lines.append(f"## {vtitle}")
        lines.append("")
        if filtered_rows:
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
            for rank, r in enumerate(filtered_rows, 1):
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
        else:
            lines.append("*No matching models.*")
        lines.append("")

    # Re-write report.md with view sections appended
    if views:
        atomic_write(report_path, "\n".join(lines))

    # Write views index for admin API (always write, even if empty, to clear stale views)
    atomic_write(results_dir / "_leaderboard_views.json",
                 json.dumps({"views": view_names}))

    logger.info("Report updated: %d completed, %d queued, %d failed",
                counts.get("COMPLETED", 0), counts.get("QUEUED", 0),
                counts.get("FAILED", 0))

    return completed




# ---------------------------------------------------------------------------
# Admin cache + report formatting (merged from leaderboard_admin.py in v4.0)
# ---------------------------------------------------------------------------


def write_admin_cache(results_dir: Path, ideas: dict, cfg: dict):
    """Write pre-aggregated _admin_cache.json for instant admin panel access."""
    now = time.time()

    # Nodes
    raw_hb = _read_all_heartbeats(results_dir, stale_seconds=600)
    heartbeats = []
    for hb in raw_hb:
        age = now - hb.get("epoch", 0)
        status = "online" if age <= 120 else ("degraded" if age <= 300 else "offline")
        heartbeats.append({**hb, "status": status, "heartbeat_age_sec": round(age, 1)})

    # Queue
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
        raw = idea.get("raw", "")
        _cat_m = re.search(r"\*\*Category\*\*:\s*(.+)", raw)
        _par_m = re.search(r"\*\*Parent\*\*:\s*(.+)", raw)
        _hyp_m = re.search(r"\*\*Hypothesis\*\*:\s*(.+)", raw)
        queue_items.append({
            "idea_id": idea_id,
            "title": idea.get("title", ""),
            "priority": idea.get("priority", "medium"),
            "status": idea_status,
            "config": idea.get("config", {}),
            "sweep_parent": idea.get("_sweep_parent"),
            "category": _cat_m.group(1).strip() if _cat_m else "architecture",
            "parent": _par_m.group(1).strip() if _par_m else "none",
            "hypothesis": _hyp_m.group(1).strip() if _hyp_m else "",
        })

    # Alerts
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
                    "type": "failure", "idea_id": entry.name,
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
        if round(usage.free / (1024 ** 3), 1) < 50:
            alerts.append({"type": "low_disk",
                           "disk_free_gb": round(usage.free / (1024 ** 3), 1)})
    except Exception:
        pass

    cache = {
        "nodes": {"heartbeats": heartbeats, "local_gpus": []},
        "queue": {"items": queue_items, "counts": all_statuses,
                  "total_all": sum(all_statuses.values())},
        "alerts": {"alerts": alerts, "count": len(alerts)},
        "epoch": now,
    }
    admin_cache_path = orze_path(cfg, "state", "admin_cache.json")
    atomic_write(admin_cache_path, json.dumps(cache, default=str))


def format_report_text(data: dict) -> str:
    """Format a periodic report summary for notifications."""
    c, f, a, q = (data.get(k, 0) for k in
                   ("completed", "failed", "active_count", "queued"))
    title = data.get("title", "Report")
    metric = data.get("metric_name", "score")
    board = data.get("leaderboard", [])
    machines = data.get("machines", [])

    lines = [title, f"{c} completed | {f} failed | {a} active | {q} queued", ""]

    if machines:
        lines.append("Machines:")
        for m in machines:
            lines.append(f"  {m.get('host','?')}: "
                         f"{m.get('gpus_busy',0)}/{m.get('gpus_total',0)} GPUs, "
                         f"{m.get('utilization','?')}% util")
        lines.append("")

    if board:
        lines.append(f"Top {len(board)} ({metric}):")
        for i, entry in enumerate(board, 1):
            val = entry.get("value")
            val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
            lines.append(f"  #{i} {entry.get('id','?')}: {val_str} "
                         f"{entry.get('title','')[:25]}")

    return "\n".join(lines)

# Legacy alias preserved for existing callers
_format_report_text = format_report_text


# ---------------------------------------------------------------------------
# NotificationProcessor (merged from engine/reporter.py in v4.0)
# ---------------------------------------------------------------------------

class NotificationProcessor:
    """Fires notifications for finished experiments.

    Owns plateau-detection counters and periodic-report timing so that
    the orchestrator can persist / restore them across restarts.
    """

    def __init__(self, results_dir: Path, cfg: dict, lake=None):
        self.results_dir = results_dir
        self.cfg = cfg
        self.lake = lake
        self._best_idea_id = None  # Optional[str]
        self._completions_since_best: int = 0
        self._plateau_notified: bool = False
        self._last_report_notify: float = 0.0

    def load_state(self, state: dict):
        """Restore persisted state from state.json."""
        self._best_idea_id = state.get("best_idea_id")
        self._completions_since_best = state.get("completions_since_best", 0)
        self._plateau_notified = state.get("plateau_notified", False)

    def get_state(self) -> dict:
        """Return state dict for persistence."""
        return {
            "best_idea_id": self._best_idea_id,
            "completions_since_best": self._completions_since_best,
            "plateau_notified": self._plateau_notified,
        }

    def process(self, finished: list, completed_rows: list, ideas: dict,
                counts: dict, active_count: int,
                save_config_hash_fn, build_machine_status_fn):
        """Fire notifications for finished experiments. Never raises."""
        try:
            cfg = self.cfg
            ncfg = cfg.get("notifications") or {}
            if not ncfg.get("enabled", False):
                logger.debug("Notifications disabled")
                return
            if not finished:
                return

            logger.info("Processing notifications for %d finished items",
                        len(finished))
            primary = cfg["report"].get("primary_metric", "test_accuracy")

            # Build rank lookup and top-10 leaderboard
            rank_lookup, leaderboard = {}, []
            for rank, r in enumerate(completed_rows, 1):
                rank_lookup[r["id"]] = rank
                if rank <= 10:
                    leaderboard.append({"id": r["id"],
                                        "title": r.get("title", r["id"]),
                                        "value": r.get("primary_val")})

            view_lbs = self._build_view_leaderboards(cfg)
            row_lookup = {r["id"]: r for r in completed_rows}

            for idea_id, gpu in finished:
                self._notify_finished(
                    idea_id, gpu, cfg, primary, row_lookup, rank_lookup,
                    leaderboard, view_lbs, ideas, save_config_hash_fn)

            # New best detection + plateau tracking
            new_best = self._check_new_best(
                completed_rows, primary, leaderboard, view_lbs, cfg)
            n_completed = self._count_completed_in_batch(finished)
            if new_best:
                self._completions_since_best = 0
                self._plateau_notified = False
            else:
                self._completions_since_best += n_completed

            self._check_plateau(completed_rows, cfg)
            self._periodic_report(ncfg, cfg, primary, counts, active_count,
                                  leaderboard, view_lbs, build_machine_status_fn)
        except Exception as e:
            logger.warning("Notification processing error: %s", e)

    # -- internal helpers ------------------------------------------------

    def _build_view_leaderboards(self, cfg: dict) -> dict:
        view_leaderboards = {}
        for view in (cfg.get("report", {}).get("views") or []):
            vname = view.get("name")
            if not vname:
                continue
            vpath = self.results_dir / f"_leaderboard_{vname}.json"
            if not vpath.exists():
                continue
            try:
                vdata = json.loads(vpath.read_text(encoding="utf-8"))
                vtop = [{"id": e.get("idea_id", "?"), "title": e.get("title", ""),
                         "value": e.get("metric_value")}
                        for e in (vdata.get("top") or [])[:10]]
                if vtop:
                    view_leaderboards[vname] = {
                        "title": vdata.get("title", vname), "entries": vtop}
            except (json.JSONDecodeError, OSError):
                pass
        return view_leaderboards

    def _notify_finished(self, idea_id, gpu, cfg, primary, row_lookup,
                         rank_lookup, leaderboard, view_lbs, ideas,
                         save_config_hash_fn):
        m_path = self.results_dir / idea_id / "metrics.json"
        if not m_path.exists():
            return
        try:
            m = json.loads(m_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return

        status = m.get("status", "UNKNOWN")
        title = ideas.get(idea_id, {}).get("title", idea_id)

        if status == "COMPLETED":
            self._notify_completed(idea_id, title, m, cfg, primary,
                                   row_lookup, rank_lookup,
                                   leaderboard, view_lbs)
        elif status == "FAILED":
            error_msg = m.get("error", "unknown")
            # Suppress notifications for config/argparse errors (exit code 2)
            # and fast crashes (<10s, typically import errors). These are
            # research-agent-generated junk, not worth spamming Telegram.
            is_config_error = "code 2" in error_msg or "code 1" in error_msg
            training_time = m.get("training_time", 999)
            if is_config_error and training_time < 10:
                logger.info("Suppressed notification for %s: config error (%s)",
                            idea_id, error_msg)
            else:
                notify("failed", {"idea_id": idea_id, "title": title,
                                  "error": error_msg,
                                  "leaderboard": leaderboard,
                                  "view_leaderboards": view_lbs}, cfg)

        if self.lake and status in ("COMPLETED", "FAILED"):
            self._archive_to_lake(idea_id, status, ideas, cfg)

        if status == "COMPLETED":
            try:
                rp = self.results_dir / idea_id / "resolved_config.yaml"
                if rp.exists():
                    rcfg = yaml.safe_load(rp.read_text(encoding="utf-8")) or {}
                    save_config_hash_fn(idea_id, rcfg)
            except Exception as exc:
                logger.debug("Config hash save failed for %s: %s",
                             idea_id, exc)

    def _notify_completed(self, idea_id, title, m, cfg, primary,
                          row_lookup, rank_lookup, leaderboard, view_lbs):
        row = row_lookup.get(idea_id, {})
        metric_val = row.get("primary_val") or m.get(primary)
        if metric_val is None:
            eval_file = cfg.get("eval_output", "eval_report.json")
            eval_path = self.results_dir / idea_id / eval_file
            if eval_path.exists():
                try:
                    ed = json.loads(eval_path.read_text(encoding="utf-8"))
                    metric_val = _resolve_primary_metric(cfg, eval_file, ed)
                except (json.JSONDecodeError, OSError,
                        KeyError, UnicodeDecodeError):
                    pass

        if metric_val is None:
            logger.warning(
                "Notification for %s has metric_val=None "
                "(row_pv=%s, m.get(%s)=%s, eval_exists=%s)",
                idea_id, row_lookup.get(idea_id, {}).get("primary_val"),
                primary, m.get(primary),
                (self.results_dir / idea_id /
                 cfg.get("eval_output", "eval_report.json")).exists())

        t_time = m.get("training_time") or None
        fmt_val = (f"{metric_val:.4f}"
                   if isinstance(metric_val, (int, float)) else metric_val)
        rank = rank_lookup.get(idea_id, None)

        # notify_top_n: only send "completed" notifications for top-N results.
        # Default 0 = notify all (backward compat). Set in orze.yaml:
        #   notifications:
        #     notify_top_n: 20
        top_n = (cfg.get("notifications") or {}).get("notify_top_n", 0)
        summary_only = (top_n > 0 and isinstance(rank, int) and rank > top_n)

        notify("completed", {
            "idea_id": idea_id, "title": title,
            "metric_name": primary, "metric_value": fmt_val,
            "training_time": t_time,
            "rank": rank if rank is not None else "?",
            "leaderboard": leaderboard,
            "view_leaderboards": view_lbs,
            "summary_only": summary_only,
        }, cfg)

    def _archive_to_lake(self, idea_id, status, ideas, cfg):
        try:
            idea_data = ideas.get(idea_id, {})
            # If the in-memory ideas dict doesn't have this idea (common:
            # ideas.md was wiped after ingestion), preserve the row that's
            # already in the lake rather than blanking config/raw_markdown.
            # Previously INSERT OR REPLACE would overwrite valid config with
            # empty strings, orphaning the idea on any retry.
            existing = None
            if not idea_data:
                try:
                    existing = self.lake.get(idea_id) if hasattr(
                        self.lake, "get") else None
                except Exception:
                    existing = None
            config_yaml = ""
            raw_md = idea_data.get("raw", "") if idea_data else (
                (existing or {}).get("raw_markdown", "") if existing else "")
            if idea_data.get("config"):
                config_yaml = yaml.dump(idea_data["config"],
                                        default_flow_style=False)
            elif existing and existing.get("config"):
                # Reuse stored config so we don't wipe it on status updates.
                config_yaml = existing["config"]
            eval_metrics = {}
            eval_file = cfg.get("eval_output", "eval_report.json")
            eval_path = self.results_dir / idea_id / eval_file
            if eval_path.exists():
                try:
                    ed = json.loads(eval_path.read_text(encoding="utf-8"))
                    em = ed.get("metrics", {})
                    for col in cfg.get("report", {}).get("columns", []):
                        src, key = col.get("source", ""), col.get("key", "")
                        if ":" in src:
                            src_file, json_path = src.split(":", 1)
                            if src_file == eval_file:
                                val = deep_get(ed, json_path)
                                if val is not None:
                                    eval_metrics[key] = val
                        elif key and key in em:
                            eval_metrics[key] = em[key]
                except (json.JSONDecodeError, OSError):
                    pass
            # Fallback: read metrics.json directly (flat format from train.py)
            if not eval_metrics:
                metrics_path = self.results_dir / idea_id / "metrics.json"
                if metrics_path.exists():
                    try:
                        md = json.loads(metrics_path.read_text(encoding="utf-8"))
                        for k, v in md.items():
                            if isinstance(v, (int, float)) and k != "num_eval_tasks":
                                eval_metrics[k] = v
                    except (json.JSONDecodeError, OSError):
                        pass

            def _raw_field(field):
                match = re.search(
                    rf"\*\*{re.escape(field)}\*\*:\s*(.+)", raw_md)
                return match.group(1).strip() if match else None

            self.lake.insert(
                idea_id, idea_data.get("title", idea_id),
                config_yaml, raw_md,
                eval_metrics=eval_metrics or None,
                status=status.lower(),
                priority=idea_data.get("priority", "medium"),
                category=_raw_field("Category"),
                parent=_raw_field("Parent"),
                hypothesis=_raw_field("Hypothesis"),
                approach_family=idea_data.get("approach_family", _raw_field("Approach Family") or "other"))
        except Exception as exc:
            logger.warning("Failed to archive %s to lake: %s", idea_id, exc)

    def _check_new_best(self, completed_rows, primary, leaderboard,
                        view_lbs, cfg) -> bool:
        if not completed_rows:
            return False
        current_best = completed_rows[0]["id"]
        fired = False
        if (self._best_idea_id is not None
                and current_best != self._best_idea_id):
            best_val = completed_rows[0].get("primary_val")
            # F14: champion-promotion guard — verify + z-score check before
            # firing new_best. If blocked, we DO NOT update self._best_idea_id
            # (caller keeps the prior best) and enqueue an audit idea.
            if isinstance(best_val, (int, float)):
                try:
                    from orze.engine.champion_guard import (
                        check_promotion, create_audit_idea,
                    )
                    def _make_audit(aid, sid, payload):
                        if getattr(self, "_lake", None) is not None:
                            create_audit_idea(self._lake, aid, sid, payload)
                    allow, info = check_promotion(
                        self.results_dir, current_best, float(best_val), cfg,
                        notify_fn=lambda k, p, c: notify(k, p, c),
                        create_audit_idea_fn=_make_audit,
                    )
                    if not allow:
                        logger.warning(
                            "champion_guard blocked promotion of %s "
                            "(claimed=%.4f verified=%s z=%s)",
                            current_best, float(best_val),
                            info.get("verified"), info.get("z"),
                        )
                        return False
                except Exception as e:  # pragma: no cover - defensive
                    logger.debug("champion_guard skipped: %s", e)
            fmt = (f"{best_val:.4f}"
                   if isinstance(best_val, (int, float)) else best_val)
            # Find previous best value for delta display
            prev_val = None
            for r in completed_rows[1:]:
                if r["id"] == self._best_idea_id:
                    prev_val = r.get("primary_val")
                    break
            prev_fmt = (f"{prev_val:.4f}"
                        if isinstance(prev_val, (int, float)) else prev_val)
            notify("new_best", {
                "idea_id": current_best,
                "title": completed_rows[0]["title"],
                "metric_name": primary, "metric_value": fmt,
                "prev_best_id": self._best_idea_id,
                "prev_best_val": prev_fmt,
                "leaderboard": leaderboard,
                "view_leaderboards": view_lbs,
            }, cfg)
            fired = True
        self._best_idea_id = current_best
        return fired

    def _count_completed_in_batch(self, finished: list) -> int:
        n = 0
        for idea_id, _ in finished:
            mp = self.results_dir / idea_id / "metrics.json"
            if not mp.exists():
                continue
            try:
                if json.loads(mp.read_text(encoding="utf-8")
                              ).get("status") == "COMPLETED":
                    n += 1
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue
        return n

    def _check_plateau(self, completed_rows, cfg):
        threshold = cfg.get("plateau_threshold", 50)
        if (threshold > 0
                and self._completions_since_best >= threshold
                and not self._plateau_notified):
            best_score = (completed_rows[0].get("primary_val")
                          if completed_rows else None)
            notify("plateau", {
                "message": (f"No improvement in {self._completions_since_best}"
                            f" ideas. Best: {best_score}"
                            f" ({self._best_idea_id})"),
                "best_id": self._best_idea_id,
                "since_best": self._completions_since_best,
                "threshold": threshold,
            }, cfg)
            self._plateau_notified = True

    def _periodic_report(self, ncfg, cfg, primary, counts, active_count,
                         leaderboard, view_lbs, build_machine_status_fn):
        interval = ncfg.get("report_interval", 0)
        if interval <= 0:
            return
        if time.time() - self._last_report_notify < interval:
            return
        notify("report", {
            "title": cfg["report"].get("title", "Report"),
            "completed": counts.get("COMPLETED", 0),
            "failed": counts.get("FAILED", 0),
            "active_count": active_count,
            "queued": counts.get("QUEUED", 0),
            "metric_name": primary,
            "leaderboard": leaderboard,
            "view_leaderboards": view_lbs,
            "machines": build_machine_status_fn(),
        }, cfg)
        self._last_report_notify = time.time()
