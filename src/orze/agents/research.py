#!/usr/bin/env python3
"""orze research agent: generic LLM-powered idea generator.

Ships with orze. Works with any LLM backend (Gemini, OpenAI, Anthropic, local).
Handles all the boilerplate: reading results, building context, formatting ideas,
appending to ideas.md. You just configure which LLM to call.

Supports multiple backends out of the box:
  - gemini   (GEMINI_API_KEY or --api-key)
  - openai   (OPENAI_API_KEY or --api-key)
  - anthropic (ANTHROPIC_API_KEY or --api-key)
  - ollama   (local, no key needed)
  - custom   (--endpoint URL for any OpenAI-compatible API)

Usage:
    # Gemini
    GEMINI_API_KEY=... python orze/research_agent.py -c orze.yaml --backend gemini

    # OpenAI
    OPENAI_API_KEY=... python orze/research_agent.py -c orze.yaml --backend openai

    # Local ollama
    python orze/research_agent.py -c orze.yaml --backend ollama --model llama3

    # Any OpenAI-compatible endpoint
    python orze/research_agent.py -c orze.yaml --backend custom --endpoint http://localhost:8080/v1

    # In orze.yaml:
    roles:
      research_gemini:
        mode: script
        script: orze/research_agent.py
        args: ["-c", "orze.yaml", "--backend", "gemini", "--cycle", "{cycle}",
               "--ideas-md", "{ideas_file}", "--results-dir", "{results_dir}"]
        timeout: 600
        cooldown: 400
        env:
          GEMINI_API_KEY: "your-key"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sqlite3
import sys
import time
import urllib.request
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("orze.research_agent")


# ---------------------------------------------------------------------------
#  Context gathering (generic — reads orze results)
# ---------------------------------------------------------------------------

def load_status(results_dir: Path) -> dict:
    """Load status.json written by the orchestrator."""
    status_path = results_dir / "status.json"
    if status_path.exists():
        try:
            return json.loads(status_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def load_full_leaderboard(results_dir: Path) -> Tuple[list, str]:
    """Load full leaderboard from _leaderboard.json (richer than status.json).

    Returns (entries, metric_name). Falls back to status.json top_results.
    """
    lb_path = results_dir / "_leaderboard.json"
    if lb_path.exists():
        try:
            data = json.loads(lb_path.read_text(encoding="utf-8"))
            return data.get("top", []), data.get("metric", "")
        except (json.JSONDecodeError, OSError):
            pass
    # Fallback to status.json
    status = load_status(results_dir)
    return status.get("top_results", []), ""


def analyze_failures(results_dir: Path,
                     max_scan: int = 2000) -> Dict[str, Any]:
    """Scan results for failures, categorize by error pattern.

    Only scans the last `max_scan` idea dirs (by name) to avoid slow
    filesystem walks on large result sets. This captures recent failure
    patterns which are most relevant for idea generation.

    Returns dict with:
        total: int
        patterns: list of {pattern, count, example_ids}
        recent: list of {idea_id, error} (last 5)
    """
    errors: List[Tuple[str, str]] = []  # (idea_id, error_msg)
    if not results_dir.exists():
        return {"total": 0, "patterns": [], "recent": []}

    # Collect idea dirs without stat-ing each entry (os.scandir is faster)
    idea_names = []
    try:
        with os.scandir(results_dir) as it:
            for entry in it:
                if entry.is_dir(follow_symlinks=False) and entry.name.startswith("idea-"):
                    idea_names.append(entry.name)
    except OSError:
        return {"total": 0, "patterns": [], "recent": []}
    idea_names.sort()
    # Only scan the tail for efficiency on large result sets
    scan_names = idea_names[-max_scan:] if len(idea_names) > max_scan else idea_names

    for name in scan_names:
        metrics_path = results_dir / name / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue
        if data.get("status") not in ("FAILED", "ERROR"):
            continue
        raw = data.get("error") or data.get("traceback") or data.get("status", "unknown")
        errors.append((name, str(raw).split("\n")[0][:200]))

    if not errors:
        return {"total": 0, "patterns": [], "recent": []}

    # Normalize errors into patterns by stripping variable parts
    def _normalize(msg: str) -> str:
        msg = re.sub(r"idea-[a-z0-9]+", "idea-XXX", msg)
        msg = re.sub(r"Unknown \w+: \S+", "Unknown <type>: <name>", msg)
        msg = re.sub(r"Stalled \(no output for \d+m\)", "Stalled (hung process)", msg)
        msg = re.sub(r"\d+\.\d+", "N", msg)
        return msg.strip()

    pattern_counter: Counter = Counter()
    pattern_examples: Dict[str, List[str]] = defaultdict(list)
    for idea_id, msg in errors:
        pat = _normalize(msg)
        pattern_counter[pat] += 1
        if len(pattern_examples[pat]) < 3:
            pattern_examples[pat].append(idea_id)

    patterns = [
        {"pattern": pat, "count": cnt, "example_ids": pattern_examples[pat]}
        for pat, cnt in pattern_counter.most_common(10)
    ]

    recent = [{"idea_id": iid, "error": msg} for iid, msg in errors[-5:]]
    return {"total": len(errors), "patterns": patterns, "recent": recent}


def _find_best_lake_metric(conn: sqlite3.Connection,
                           primary_metric: str) -> str:
    """Find the best available metric in the lake.

    Tries the primary_metric first, then strips prefixes to find alternatives.
    E.g., if 'my_adjusted_auc' isn't in the lake, tries 'my_auc'.
    """
    if primary_metric:
        count = conn.execute(
            "SELECT COUNT(*) FROM ideas "
            "WHERE json_extract(eval_metrics, ?) IS NOT NULL",
            (f"$.{primary_metric}",),
        ).fetchone()[0]
        if count > 0:
            return primary_metric

    # Try common suffixes/variants
    candidates = []
    if primary_metric:
        # Strip "adjusted_" prefix: e.g. foo_adjusted_auc -> foo_auc
        candidates.append(re.sub(r"_adjusted_", "_", primary_metric))
        # Just "auc" or the last part
        parts = primary_metric.split("_")
        if len(parts) > 1:
            candidates.append("_".join(parts[1:]))  # drop first prefix

    # Find which has most data
    best, best_count = "", 0
    for cand in candidates:
        if not cand:
            continue
        count = conn.execute(
            "SELECT COUNT(*) FROM ideas "
            "WHERE json_extract(eval_metrics, ?) IS NOT NULL",
            (f"$.{cand}",),
        ).fetchone()[0]
        if count > best_count:
            best, best_count = cand, count

    if best:
        logger.info("Lake metric fallback: %s -> %s (%d ideas)",
                     primary_metric, best, best_count)
    return best


def load_lake_summary(lake_db_path: Path, primary_metric: str) -> Optional[dict]:
    """Query idea_lake.db for historical stats and performance patterns.

    Returns None if lake doesn't exist. Otherwise returns:
        status_counts: {status: count}
        top_from_lake: list of {idea_id, title, metric_value}
        config_patterns: {dimension: [{value, mean, count}]}
        lake_metric: str (the metric actually used)
    """
    if not lake_db_path.exists():
        return None
    try:
        conn = sqlite3.connect(str(lake_db_path), timeout=10)
        conn.row_factory = sqlite3.Row
    except Exception:
        return None

    result: Dict[str, Any] = {}
    try:
        # Status breakdown
        rows = conn.execute(
            "SELECT status, COUNT(*) as cnt FROM ideas GROUP BY status"
        ).fetchall()
        result["status_counts"] = {r["status"]: r["cnt"] for r in rows}

        # Find best available metric
        lake_metric = _find_best_lake_metric(conn, primary_metric)
        result["lake_metric"] = lake_metric

        if lake_metric:
            metric_path = f"$.{lake_metric}"
            top_rows = conn.execute(
                "SELECT idea_id, title, "
                "json_extract(eval_metrics, ?) as metric_val "
                "FROM ideas "
                "WHERE json_extract(eval_metrics, ?) IS NOT NULL "
                "ORDER BY json_extract(eval_metrics, ?) DESC LIMIT 10",
                (metric_path, metric_path, metric_path),
            ).fetchall()
            result["top_from_lake"] = [
                {"idea_id": r["idea_id"], "title": r["title"],
                 "metric_value": r["metric_val"]}
                for r in top_rows
            ]

            result["config_patterns"] = _analyze_config_dimensions(
                conn, lake_metric
            )

    except Exception as e:
        logger.warning("Lake query failed: %s", e)
    finally:
        conn.close()

    return result


def _extract_flat_keys(config: dict, prefix: str = "") -> Dict[str, Any]:
    """Flatten a nested config dict into dot-separated keys.

    Only keeps leaf scalar values (str, int, float, bool).
    Limits depth to 2 levels to avoid noise.
    """
    result = {}
    for key, val in config.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict) and not prefix:
            result.update(_extract_flat_keys(val, full_key))
        elif not isinstance(val, (dict, list)) and val is not None:
            result[full_key] = val
    return result


def _analyze_config_dimensions(
    conn: sqlite3.Connection, primary_metric: str, max_samples: int = 5000
) -> Dict[str, list]:
    """Group completed ideas by config keys, compute per-value stats.

    Uses pre-computed config_summary column for O(1) key access.
    Limits analysis to top max_samples ideas for scalability.
    """
    metric_path = f"$.{primary_metric}"
    # Sample the best ideas to find winning patterns
    rows = conn.execute(
        "SELECT config_summary, "
        "json_extract(eval_metrics, ?) as metric_val "
        "FROM ideas "
        "WHERE json_extract(eval_metrics, ?) IS NOT NULL "
        "ORDER BY metric_val DESC LIMIT ?",
        (metric_path, metric_path, max_samples),
    ).fetchall()

    if not rows:
        return {}

    # Collect all config dimensions and their values with metrics
    dim_data: Dict[str, Dict[str, List[float]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for r in rows:
        metric_val = r["metric_val"]
        if metric_val is None:
            continue

        cs = None
        if r["config_summary"]:
            try:
                cs = json.loads(r["config_summary"])
            except (json.JSONDecodeError, TypeError):
                pass

        if not cs:
            continue

        for key, val in cs.items():
            if val is None or isinstance(val, (dict, list)):
                continue
            dim_data[key][str(val)].append(float(metric_val))

    # For each dimension, compute mean + count per value, sorted by mean desc
    # Only keep dimensions where at least 2 values have n >= 3
    patterns: Dict[str, list] = {}
    for dim, val_map in dim_data.items():
        if len(val_map) < 2:
            continue
        entries = []
        for val, metrics in val_map.items():
            if len(metrics) < 3:
                continue  # skip values with too few samples
            entries.append({
                "value": val,
                "mean": round(sum(metrics) / len(metrics), 4),
                "count": len(metrics),
            })
        if len(entries) < 2:
            continue  # need at least 2 values to compare
        entries.sort(key=lambda x: -x["mean"])
        patterns[dim] = entries[:5]

    return patterns


def get_existing_idea_ids(ideas_path: Path) -> List[str]:
    """Parse ideas.md and return list of existing idea IDs."""
    if not ideas_path.exists():
        return []
    text = ideas_path.read_text(encoding="utf-8")
    return [m.group(1) for m in re.finditer(r"^## (idea-[a-z0-9]+):", text, re.MULTILINE)]


def parse_idea_configs(ideas_path: Path) -> Dict[str, dict]:
    """Parse ideas.md and return {idea_id: config_dict} for ideas with YAML."""
    if not ideas_path.exists():
        return {}
    text = ideas_path.read_text(encoding="utf-8")
    ideas = {}
    pattern = re.compile(r"^## (idea-[a-z0-9]+):\s*(.+?)$", re.MULTILINE)
    matches = list(pattern.finditer(text))
    for i, m in enumerate(matches):
        idea_id = m.group(1)
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        block = text[start:end]
        yaml_match = re.search(r"```ya?ml\s*\n(.*?)```", block, re.DOTALL)
        if yaml_match:
            try:
                config = yaml.safe_load(yaml_match.group(1))
                if isinstance(config, dict):
                    ideas[idea_id] = config
            except yaml.YAMLError:
                pass
    return ideas


def generate_idea_id(config: dict, results_dir: Path) -> str:
    """Generate a 6-char content-hash idea ID, collision-free.

    Hash = sha256(yaml.dump(config, sort_keys=True) + nonce)[:6].
    Checks results/ to avoid collisions with existing experiments.
    """
    import hashlib
    import time as _time
    raw = yaml.dump(config, sort_keys=True)
    for nonce in range(100):
        h = hashlib.sha256(f"{raw}:{nonce}".encode()).hexdigest()[:6]
        idea_id = f"idea-{h}"
        if not (results_dir / idea_id).exists():
            return idea_id
    # Fallback: timestamp-based
    return f"idea-{hashlib.sha256(f'{raw}:{_time.time()}'.encode()).hexdigest()[:6]}"


def build_context(results_dir: Path, ideas_path: Path, report_cfg: dict,
                  lake_db_path: Optional[Path] = None) -> str:
    """Build a comprehensive context string for the LLM.

    Includes: stats, full leaderboard with all metrics, categorized failure
    analysis, historical performance patterns from idea_lake.db, and example
    configs from top performers.
    """
    status = load_status(results_dir)
    existing_ids = get_existing_idea_ids(ideas_path)
    primary_metric = report_cfg.get("primary_metric", "")
    columns = report_cfg.get("columns", [])

    lines = ["# Current Research State\n"]
    lines.append(f"- Total ideas in queue: {len(existing_ids)}")
    lines.append(f"- Completed: {status.get('completed', 0)}")
    lines.append(f"- Failed: {status.get('failed', 0)}")
    lines.append(f"- In queue: {status.get('queue_depth', 0)}")
    if primary_metric:
        lines.append(f"- Primary metric: {primary_metric}")
    lines.append("")

    # --- Full leaderboard from _leaderboard.json ---
    lb_entries, lb_metric = load_full_leaderboard(results_dir)
    if lb_entries:
        lines.append("## Leaderboard (top and bottom focus)\n")
        col_labels = [c.get("label", c["key"]) for c in columns] if columns else []
        
        def _format_row(i, entry):
            idea_id = entry.get("idea_id", "?")
            title = entry.get("title", "")[:40]
            em = entry.get("eval_metrics", {})
            if col_labels and columns:
                vals = []
                for c in columns:
                    key = c["key"]
                    fmt = c.get("fmt", "")
                    val = em.get(key, entry.get(key, ""))
                    if val != "" and val is not None and fmt:
                        try:
                            val = f"{float(val):{fmt}}"
                        except (ValueError, TypeError):
                            val = str(val)
                    vals.append(str(val) if val is not None else "")
                return f"| {i} | {idea_id} | {title} | " + " | ".join(vals) + " |"
            else:
                score = em.get(primary_metric, entry.get("metric_value", "?"))
                return f"| {i} | {idea_id} | {title} | {score} |"

        if col_labels:
            header = "| # | Idea | Title | " + " | ".join(col_labels) + " |"
            sep = "|---|------|-------|" + "|".join(["---"] * len(col_labels)) + "|"
        else:
            header = "| # | Idea | Title | Score |"
            sep = "|---|------|-------|-------|"
            
        lines.append(header)
        lines.append(sep)

        # Token Optimization: Only show top 10 and bottom 5 in full detail
        # Show middle as summary to save repetitive tokens
        n_total = len(lb_entries)
        for i, entry in enumerate(lb_entries, 1):
            if i <= 10 or i > (n_total - 5):
                lines.append(_format_row(i, entry))
            elif i == 11:
                lines.append(f"| ... | ... | [{n_total-15} intermediate models omitted] | ... |")
        lines.append("")

    # --- Failure analysis ---
    failures = analyze_failures(results_dir)
    if failures["total"] > 0:
        lines.append(f"## Failure Analysis ({failures['total']} total failures)\n")
        lines.append("**Top failure patterns** (avoid generating ideas that would hit these):\n")
        for p in failures["patterns"]:
            examples = ", ".join(p["example_ids"][:2])
            lines.append(f"- **{p['count']}x**: {p['pattern']} (e.g. {examples})")
        if failures["recent"]:
            lines.append("\n**Most recent failures:**\n")
            for f in failures["recent"]:
                lines.append(f"- {f['idea_id']}: {f['error']}")
        lines.append("")

    # --- Historical stats from idea_lake.db ---
    if lake_db_path is None:
        lake_db_path = ideas_path.parent / "idea_lake.db"
    lake = load_lake_summary(lake_db_path, primary_metric)
    if lake:
        sc = lake.get("status_counts", {})
        total_historical = sum(sc.values())
        if total_historical > 0:
            lines.append(f"## Historical Stats (from {total_historical} total experiments)\n")
            parts = [f"{s}: {c}" for s, c in sorted(sc.items(), key=lambda x: -x[1])]
            lines.append("- " + ", ".join(parts))
            lines.append("")

        lake_metric = lake.get("lake_metric", primary_metric)
        top_lake = lake.get("top_from_lake", [])
        if top_lake:
            lines.append(f"## All-Time Best ({lake_metric})\n")
            for i, t in enumerate(top_lake[:5], 1):
                mv = t["metric_value"]
                if isinstance(mv, float):
                    mv = f"{mv:.4f}"
                lines.append(f"{i}. {t['idea_id']}: {mv} — {t['title'][:50]}")
            lines.append("")

        config_patterns = lake.get("config_patterns", {})
        if config_patterns:
            lines.append("## Performance by Config Dimension\n")
            lines.append("Shows which config values correlate with better results.\n")
            for dim, entries in sorted(config_patterns.items()):
                if not entries:
                    continue
                best = entries[0]
                worst = entries[-1] if len(entries) > 1 else None
                line = f"- **{dim}**: best={best['value']} (mean {lake_metric}={best['mean']}, n={best['count']})"
                if worst and worst["value"] != best["value"]:
                    line += f", worst={worst['value']} ({worst['mean']}, n={worst['count']})"
                lines.append(line)
            lines.append("")

    # --- Example configs from top leaderboard ideas ---
    if lb_entries and ideas_path.exists():
        idea_configs = parse_idea_configs(ideas_path)
        top_ids = [e.get("idea_id") for e in lb_entries[:5] if e.get("idea_id")]
        shown = 0
        for tid in top_ids:
            if tid in idea_configs and shown < 2:
                lines.append(f"## Example Config ({tid} — top performer)\n")
                lines.append("```yaml")
                config_yaml = yaml.dump(idea_configs[tid],
                                        default_flow_style=False, sort_keys=False)
                lines.append(config_yaml.rstrip())
                lines.append("```")
                lines.append("")
                shown += 1
        if shown == 0:
            for iid, cfg in list(idea_configs.items())[-2:]:
                lines.append(f"## Example Config ({iid})\n")
                lines.append("```yaml")
                config_yaml = yaml.dump(cfg, default_flow_style=False,
                                        sort_keys=False)
                lines.append(config_yaml.rstrip())
                lines.append("```")
                lines.append("")

    # --- Existing idea IDs (dedup hint) ---
    if existing_ids:
        tail = existing_ids[-50:]
        lines.append(f"## Existing Ideas (last {len(tail)} of {len(existing_ids)} — avoid duplicates)\n")
        lines.append(", ".join(tail))
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
#  Idea formatting and appending (generic — matches orze format)
# ---------------------------------------------------------------------------

def format_idea_markdown(idea_id: str, title: str, hypothesis: str,
                         config: dict, priority: str = "high",
                         category: str = "architecture",
                         parent: str = "none",
                         cycle: int = 0) -> str:
    """Format a single idea as markdown for appending to ideas.md."""
    lines = [
        f"\n## {idea_id}: {title}",
        f"- **Priority**: {priority}",
        f"- **Category**: {category}",
        f"- **Parent**: {parent}",
        f"- **Research Cycle**: {cycle}",
        f"- **Hypothesis**: {hypothesis}",
        "- **Config overrides**:",
        "  ```yaml",
    ]
    config_yaml = yaml.dump(config, default_flow_style=False, sort_keys=False)
    for line in config_yaml.splitlines():
        lines.append(f"  {line}")
    lines.append("  ```")
    lines.append("")
    return "\n".join(lines)


def append_ideas_to_md(ideas_md: list, ideas_path: Path,
                       results_dir: Optional[Path] = None) -> int:
    """Append formatted idea markdown strings to ideas.md. Returns count.

    Acquires filesystem lock to prevent race with orchestrator's
    ideas.md consumption (which wipes the file after ingesting to SQLite).
    """
    if not ideas_md:
        return 0
    from orze.core.fs import _fs_lock, _fs_unlock
    # Use same lock path as orchestrator: results_dir / ".ideas_md.lock"
    if results_dir:
        lock_dir = results_dir / ".ideas_md.lock"
    else:
        lock_dir = ideas_path.parent / ".ideas_md.lock"
    locked = _fs_lock(lock_dir, stale_seconds=60)
    try:
        with open(ideas_path, "a", encoding="utf-8") as f:
            f.write("\n")
            for md in ideas_md:
                f.write(md)
                f.write("\n")
    finally:
        if locked:
            _fs_unlock(lock_dir)
    return len(ideas_md)


# ---------------------------------------------------------------------------
#  LLM response parsing (generic — extracts ideas from LLM output)
# ---------------------------------------------------------------------------

def parse_llm_ideas(response: str, results_dir: Path, cycle: int) -> list:
    """Parse LLM response into structured ideas.

    Expects the LLM to return ideas as JSON array or markdown.
    Tries JSON first, falls back to markdown parsing.
    IDs are 6-char content hashes of the config YAML.

    Each idea needs: title, hypothesis, config (dict).
    Optional: priority, category, parent.
    """
    ideas = []

    # Try JSON array first
    json_ideas = _try_parse_json(response)
    if json_ideas:
        for i, item in enumerate(json_ideas):
            if not isinstance(item, dict):
                continue
            if not item.get("title") or not item.get("config"):
                continue
            idea_id = generate_idea_id(item["config"], results_dir)
            ideas.append({
                "idea_id": idea_id,
                "title": item["title"],
                "hypothesis": item.get("hypothesis", item.get("title")),
                "config": item["config"],
                "priority": item.get("priority", "high"),
                "category": item.get("category", "architecture"),
                "parent": item.get("parent", "none"),
                "cycle": cycle,
            })
        return ideas

    # Fallback: try to find YAML blocks in markdown
    pattern = re.compile(
        r"##\s*(?:idea-[a-z0-9]+:\s*)?(.+?)$\s*"
        r"(?:.*?hypothesis[:\s]*(.+?)$)?\s*"
        r"```ya?ml\s*\n(.*?)```",
        re.MULTILINE | re.DOTALL | re.IGNORECASE,
    )
    for i, m in enumerate(pattern.finditer(response)):
        title = m.group(1).strip()
        hypothesis = (m.group(2) or title).strip()
        try:
            config = yaml.safe_load(m.group(3))
        except yaml.YAMLError:
            continue
        if not isinstance(config, dict):
            continue
        idea_id = generate_idea_id(config, results_dir)
        ideas.append({
            "idea_id": idea_id,
            "title": title,
            "hypothesis": hypothesis,
            "config": config,
            "priority": "high",
            "category": "architecture",
            "parent": "none",
            "cycle": cycle,
        })

    return ideas


def _try_parse_json(text: str) -> Optional[list]:
    """Try to extract a JSON array from text, handling brackets inside strings."""
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    in_string = False
    escape = False
    end = start
    for i in range(start, len(text)):
        c = text[i]
        if escape:
            escape = False
            continue
        if c == "\\":
            if in_string:
                escape = True
            continue
        if c == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if c == "[":
            depth += 1
        elif c == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break
    if depth != 0:
        return None
    try:
        result = json.loads(text[start:end])
        return result if isinstance(result, list) else None
    except json.JSONDecodeError:
        return None


# ---------------------------------------------------------------------------
#  LLM backends
# ---------------------------------------------------------------------------

def _post_json(url: str, payload: dict, headers: dict,
               timeout: int = 120) -> dict:
    """POST JSON and return parsed response."""
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, method="POST", headers=headers)
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


def call_gemini(prompt: str, api_key: str,
                model: str = "gemini-2.5-flash",
                max_tokens: int = 8192,
                web_search: bool = False) -> str:
    """Call Gemini API. Tries multiple models on failure.

    Args:
        web_search: Enable Google Search grounding so Gemini can fetch
            live web results alongside its own knowledge.
    """
    models = [model, "gemini-3-flash-preview", "gemini-2.5-flash"]
    # Deduplicate while preserving order
    seen = set()
    unique_models = []
    for m in models:
        if m not in seen:
            seen.add(m)
            unique_models.append(m)

    for m in unique_models:
        try:
            url = (f"https://generativelanguage.googleapis.com/v1beta/"
                   f"models/{m}:generateContent?key={api_key}")
            payload = {
                "contents": [{"parts": [{"text": prompt}]}],
                "generationConfig": {"maxOutputTokens": max_tokens},
            }
            if web_search:
                payload["tools"] = [{"google_search": {}}]
            result = _post_json(url, payload, {"Content-Type": "application/json"})
            parts = (result.get("candidates", [{}])[0]
                     .get("content", {}).get("parts", []))
            text = "\n".join(p.get("text", "") for p in (parts or []) if p.get("text"))
            if text:
                logger.info("Gemini (%s) returned %d chars", m, len(text))
                return text
            logger.warning("Gemini (%s) returned empty response", m)
        except Exception as e:
            logger.warning("Gemini (%s) failed: %s", m, e)
    return ""


def call_openai(prompt: str, api_key: str,
                model: str = "gpt-4o",
                max_tokens: int = 8192,
                endpoint: str = "https://api.openai.com/v1") -> str:
    """Call OpenAI-compatible API (OpenAI, Azure, vLLM, etc.)."""
    url = f"{endpoint.rstrip('/')}/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": max_tokens,
        "temperature": 0.7,
    }
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}",
    }
    try:
        result = _post_json(url, payload, headers)
        text = result.get("choices", [{}])[0].get("message", {}).get("content", "")
        logger.info("OpenAI (%s) returned %d chars", model, len(text))
        return text
    except Exception as e:
        logger.error("OpenAI (%s) failed: %s", model, e)
        return ""


def call_anthropic(prompt: str, api_key: str,
                   model: str = "claude-sonnet-4-5-20250929",
                   max_tokens: int = 8192) -> str:
    """Call Anthropic API directly."""
    url = "https://api.anthropic.com/v1/messages"
    payload = {
        "model": model,
        "max_tokens": max_tokens,
        "messages": [{"role": "user", "content": prompt}],
    }
    headers = {
        "Content-Type": "application/json",
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    try:
        result = _post_json(url, payload, headers)
        text = "".join(
            b.get("text", "") for b in result.get("content", [])
            if b.get("type") == "text"
        )
        logger.info("Anthropic (%s) returned %d chars", model, len(text))
        return text
    except Exception as e:
        logger.error("Anthropic (%s) failed: %s", model, e)
        return ""


def call_ollama(prompt: str, model: str = "llama3",
                endpoint: str = "http://localhost:11434") -> str:
    """Call local Ollama instance."""
    url = f"{endpoint.rstrip('/')}/api/generate"
    payload = {"model": model, "prompt": prompt, "stream": False}
    try:
        result = _post_json(url, payload, {"Content-Type": "application/json"},
                            timeout=300)
        text = result.get("response", "")
        logger.info("Ollama (%s) returned %d chars", model, len(text))
        return text
    except Exception as e:
        logger.error("Ollama (%s) failed: %s", model, e)
        return ""


def call_llm(prompt: str, backend: str, api_key: str = "",
             model: str = "", endpoint: str = "") -> str:
    """Route to the appropriate LLM backend."""
    if backend == "gemini":
        key = api_key or os.environ.get("GEMINI_API_KEY", "")
        if not key:
            logger.error("GEMINI_API_KEY not set")
            return ""
        return call_gemini(prompt, key, model=model or "gemini-2.5-flash",
                          web_search=True)

    elif backend == "openai":
        key = api_key or os.environ.get("OPENAI_API_KEY", "")
        if not key:
            logger.error("OPENAI_API_KEY not set")
            return ""
        return call_openai(prompt, key, model=model or "gpt-4o",
                           endpoint=endpoint or "https://api.openai.com/v1")

    elif backend == "anthropic":
        key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        if not key:
            logger.error("ANTHROPIC_API_KEY not set")
            return ""
        return call_anthropic(prompt, key,
                              model=model or "claude-sonnet-4-5-20250929")

    elif backend == "ollama":
        return call_ollama(prompt, model=model or "llama3",
                           endpoint=endpoint or "http://localhost:11434")

    elif backend == "custom":
        # OpenAI-compatible custom endpoint
        key = api_key or os.environ.get("LLM_API_KEY", "")
        return call_openai(prompt, key, model=model or "default",
                           endpoint=endpoint)

    else:
        logger.error("Unknown backend: %s", backend)
        return ""


# ---------------------------------------------------------------------------
#  Prompt building
# ---------------------------------------------------------------------------

DEFAULT_SYSTEM_PROMPT = """\
You are a research agent for an automated ML experiment system called orze.
Your job is to analyze past results and generate new experiment ideas.

## How It Works
- You receive the full leaderboard, failure analysis, and performance patterns
- You propose new experiments as structured ideas
- Each idea needs: a title, hypothesis, and YAML config
- The system will automatically train and evaluate your ideas

## Strategy Guidelines
- **Study the leaderboard**: understand what makes top performers successful
- **Study the failures**: avoid config patterns that consistently fail
- **Use performance patterns**: config dimensions show which values correlate with
  better results — build on high-performing values, avoid low-performing ones
- **Set parent IDs**: when iterating on a successful experiment, set "parent" to
  its idea ID so the lineage is tracked
- **Balance exploitation and exploration**: ~60% ideas should refine what works,
  ~40% should try genuinely new approaches

## Output Format
Return a JSON array of ideas. Each idea is an object with:
- "title": short descriptive name (string)
- "hypothesis": why this might work, referencing evidence from the context (string)
- "config": YAML-compatible dict with experiment config (object)
- "priority": "critical" | "high" | "medium" | "low" (string, optional)
- "category": free-form label like "architecture", "hyperparameter", "loss" (string, optional)
- "parent": "none" or an existing idea ID if building on a previous idea (string, optional)

Note: idea IDs are auto-generated as 6-char content hashes (e.g. "idea-a7f3b2").
You do NOT need to assign IDs — just provide the config and orze will hash it.

Example:
```json
[
  {
    "title": "Larger learning rate with cosine schedule",
    "hypothesis": "Current best uses lr=1e-4. A 3x larger lr with cosine decay may converge faster and find a better minimum.",
    "config": {
      "model": {"type": "resnet50", "pretrained": true},
      "training": {"lr": 3e-4, "scheduler": "cosine", "epochs": 20}
    },
    "priority": "high",
    "category": "hyperparameter",
    "parent": "none"
  }
]
```

Output ONLY the JSON array, no markdown fences or extra text.
"""


def build_prompt(context: str, rules_content: str, num_ideas: int,
                 retrospection: str = "") -> str:
    """Build the full prompt for the LLM."""
    parts = [DEFAULT_SYSTEM_PROMPT]

    if retrospection:
        parts.append("## Retrospection Analysis\n")
        parts.append("The following is an automated analysis of recent experiment trends "
                     "and patterns. Use these insights to guide your idea generation.\n")
        parts.append(retrospection)
        parts.append("")

    if rules_content:
        parts.append("## Project-Specific Rules\n")
        parts.append(rules_content)
        parts.append("")

    parts.append(context)

    parts.append(f"\n## Your Task\n")
    parts.append(f"Generate exactly {num_ideas} new experiment ideas as a JSON array.")
    parts.append("Use the leaderboard, failure analysis, and performance patterns above to inform your choices.")
    parts.append("Each idea must have a unique, testable hypothesis grounded in evidence from the context.\n")

    return "\n".join(parts)


# ---------------------------------------------------------------------------
#  Main research cycle
# ---------------------------------------------------------------------------

def run_research_cycle(
    backend: str,
    cycle: int,
    ideas_path: Path,
    results_dir: Path,
    report_cfg: dict,
    num_ideas: int = 5,
    api_key: str = "",
    model: str = "",
    endpoint: str = "",
    rules_file: str = "",
    lake_db_path: Optional[Path] = None,
    dry_run: bool = False,
    retrospection_file: str = "",
) -> int:
    """Run one research cycle. Returns number of ideas generated."""
    logger.info("=" * 60)
    logger.info("RESEARCH CYCLE %d (%s)", cycle, backend)
    logger.info("=" * 60)

    # 1. Build context from results
    logger.info("Step 1: Building research context...")
    context = build_context(results_dir, ideas_path, report_cfg,
                            lake_db_path=lake_db_path)
    existing_ids = get_existing_idea_ids(ideas_path)
    logger.info("  %d existing ideas in queue, using content-hash IDs", len(existing_ids))

    # 2. Load project-specific rules if provided
    rules_content = ""
    if rules_file:
        rules_path = Path(rules_file)
        if rules_path.exists():
            rules_content = rules_path.read_text(encoding="utf-8")
            logger.info("  Loaded rules from %s (%d chars)", rules_file, len(rules_content))

    # 2b. Load retrospection output if provided
    retro_content = ""
    if retrospection_file:
        retro_path = Path(retrospection_file)
        if retro_path.exists():
            try:
                retro_content = retro_path.read_text(encoding="utf-8").strip()
                if retro_content:
                    logger.info("  Loaded retrospection from %s (%d chars)",
                                retrospection_file, len(retro_content))
            except OSError as e:
                logger.warning("  Could not read retrospection file: %s", e)

    # 3. Build prompt
    prompt = build_prompt(context, rules_content, num_ideas,
                          retrospection=retro_content)

    if dry_run:
        print("=" * 60)
        print("DRY RUN — prompt that would be sent to LLM:")
        print("=" * 60)
        print(prompt)
        print("=" * 60)
        print(f"Prompt length: {len(prompt)} chars")
        return 0

    # 4. Call LLM
    logger.info("Step 2: Calling %s (prompt: %d chars)...", backend, len(prompt))
    response = call_llm(prompt, backend, api_key=api_key, model=model,
                        endpoint=endpoint)
    if not response:
        logger.error("LLM returned empty response — aborting cycle")
        return 0

    # 5. Parse ideas from response
    logger.info("Step 3: Parsing ideas from response...")
    ideas = parse_llm_ideas(response, results_dir, cycle)
    if not ideas:
        logger.warning("Could not parse any ideas from LLM response")
        logger.debug("Response was: %s", response[:2000])
        return 0
    logger.info("  Parsed %d ideas", len(ideas))

    # 6. Format and append to ideas.md
    ideas_md = []
    for idea in ideas:
        md = format_idea_markdown(
            idea_id=idea["idea_id"],
            title=idea["title"],
            hypothesis=idea["hypothesis"],
            config=idea["config"],
            priority=idea.get("priority", "high"),
            category=idea.get("category", "architecture"),
            parent=idea.get("parent", "none"),
            cycle=cycle,
        )
        ideas_md.append(md)
        logger.info("  %s: %s", idea["idea_id"], idea["title"][:60])

    count = append_ideas_to_md(ideas_md, ideas_path, results_dir=results_dir)
    logger.info("Appended %d ideas to %s", count, ideas_path)

    logger.info("Research cycle %d complete: %d new ideas", cycle, count)
    return count


# ---------------------------------------------------------------------------
#  CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="orze research agent — generate experiment ideas with any LLM",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""\
Examples:
  # Gemini
  GEMINI_API_KEY=... python orze/research_agent.py -c orze.yaml --backend gemini

  # OpenAI
  OPENAI_API_KEY=... python orze/research_agent.py -c orze.yaml --backend openai

  # Local ollama
  python orze/research_agent.py -c orze.yaml --backend ollama --model llama3

  # Custom endpoint
  python orze/research_agent.py -c orze.yaml --backend custom --endpoint http://localhost:8080/v1

  # In orze.yaml roles:
  research_gemini:
    mode: script
    script: orze/research_agent.py
    args: ["-c", "orze.yaml", "--backend", "gemini", "--cycle", "{cycle}",
           "--ideas-md", "{ideas_file}", "--results-dir", "{results_dir}"]
    env:
      GEMINI_API_KEY: "your-key"
""",
    )

    parser.add_argument("-c", "--config", default="orze.yaml",
                        help="Path to orze.yaml (for results_dir, ideas_file, report config)")
    parser.add_argument("--backend", default="gemini",
                        choices=["gemini", "openai", "anthropic", "ollama", "custom"],
                        help="LLM backend to use (default: gemini)")
    parser.add_argument("--model", default="",
                        help="Model name (default: backend-specific)")
    parser.add_argument("--api-key", default="",
                        help="API key (default: from environment)")
    parser.add_argument("--endpoint", default="",
                        help="API endpoint URL (for custom/ollama backends)")
    parser.add_argument("--cycle", type=int, default=1,
                        help="Research cycle number")
    parser.add_argument("--num-ideas", type=int, default=5,
                        help="Number of ideas to generate (default: 5)")
    parser.add_argument("--ideas-md", default="",
                        help="Path to ideas.md (overrides orze.yaml)")
    parser.add_argument("--results-dir", default="",
                        help="Path to results dir (overrides orze.yaml)")
    parser.add_argument("--rules-file", default="",
                        help="Path to project-specific rules file")
    parser.add_argument("--lake-db", default="",
                        help="Path to idea_lake.db (default: alongside ideas.md)")
    parser.add_argument("--retrospection-file", default="",
                        help="Path to retrospection output file (auto-generated by orze)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print the prompt and exit without calling the LLM")

    args = parser.parse_args()

    # Load orze.yaml for defaults
    cfg = {}
    config_path = Path(args.config)
    if config_path.exists():
        try:
            cfg = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
        except Exception as e:
            logger.warning("Could not load %s: %s", args.config, e)

    ideas_path = Path(args.ideas_md or cfg.get("ideas_file", "ideas.md"))
    results_dir = Path(args.results_dir or cfg.get("results_dir", "results"))
    report_cfg = cfg.get("report", {})
    lake_db_path = Path(args.lake_db) if args.lake_db else None

    count = run_research_cycle(
        backend=args.backend,
        cycle=args.cycle,
        ideas_path=ideas_path,
        results_dir=results_dir,
        report_cfg=report_cfg,
        num_ideas=args.num_ideas,
        api_key=args.api_key,
        model=args.model,
        endpoint=args.endpoint,
        rules_file=args.rules_file,
        lake_db_path=lake_db_path,
        dry_run=args.dry_run,
        retrospection_file=args.retrospection_file,
    )

    sys.exit(0 if count > 0 else 1)


if __name__ == "__main__":
    main()
