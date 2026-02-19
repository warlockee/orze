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
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional

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
    """Load status.json written by farm.py."""
    status_path = results_dir / "status.json"
    if status_path.exists():
        try:
            return json.loads(status_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return {}


def load_leaderboard(results_dir: Path) -> list:
    """Load top results from status.json."""
    status = load_status(results_dir)
    return status.get("top_results", [])


def load_completed_ideas(results_dir: Path) -> Dict[str, dict]:
    """Load metrics from all completed ideas."""
    results = {}
    if not results_dir.exists():
        return results
    for idea_dir in sorted(results_dir.iterdir()):
        if not idea_dir.is_dir() or not idea_dir.name.startswith("idea-"):
            continue
        metrics_path = idea_dir / "metrics.json"
        if not metrics_path.exists():
            continue
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            results[idea_dir.name] = metrics
        except (json.JSONDecodeError, OSError):
            pass
    return results


def get_existing_idea_ids(ideas_path: Path) -> List[str]:
    """Parse ideas.md and return list of existing idea IDs."""
    if not ideas_path.exists():
        return []
    text = ideas_path.read_text(encoding="utf-8")
    return [m.group(1) for m in re.finditer(r"^## (idea-\d+):", text, re.MULTILINE)]


def parse_idea_configs(ideas_path: Path) -> Dict[str, dict]:
    """Parse ideas.md and return {idea_id: config_dict} for ideas with YAML."""
    if not ideas_path.exists():
        return {}
    text = ideas_path.read_text(encoding="utf-8")
    ideas = {}
    pattern = re.compile(r"^## (idea-\d+):\s*(.+?)$", re.MULTILINE)
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


def next_idea_id(existing_ids: List[str]) -> int:
    """Get the next available idea number."""
    nums = []
    for id_str in existing_ids:
        m = re.match(r"idea-(\d+)", id_str)
        if m:
            nums.append(int(m.group(1)))
    return max(nums) + 1 if nums else 1


def build_context(results_dir: Path, ideas_path: Path, report_cfg: dict) -> str:
    """Build a context string summarizing current research state.

    This is the generic context that any LLM can use to understand
    what's been tried and what's working.
    """
    status = load_status(results_dir)
    completed = load_completed_ideas(results_dir)
    existing_ids = get_existing_idea_ids(ideas_path)

    # Basic stats
    lines = ["# Current Research State\n"]
    lines.append(f"- Total ideas: {len(existing_ids)}")
    lines.append(f"- Completed: {status.get('completed', len(completed))}")
    lines.append(f"- Failed: {status.get('failed', 0)}")
    lines.append(f"- In queue: {status.get('queue_depth', 0)}")
    lines.append("")

    # Leaderboard
    top = status.get("top_results", [])
    primary_metric = report_cfg.get("primary_metric", "")
    columns = report_cfg.get("columns", [])
    if top:
        lines.append("## Top Results (Leaderboard)\n")
        # Build header from report columns
        col_labels = [c.get("label", c["key"]) for c in columns] if columns else []
        if col_labels:
            lines.append("| # | Idea | " + " | ".join(col_labels) + " |")
            lines.append("|---|------|" + "|".join(["---"] * len(col_labels)) + "|")
        else:
            lines.append("| # | Idea | Score |")
            lines.append("|---|------|-------|")

        for i, result in enumerate(top[:15], 1):
            idea_id = result.get("idea_id", "?")
            if col_labels and columns:
                vals = []
                for c in columns:
                    key = c["key"]
                    fmt = c.get("fmt", "")
                    val = result.get(key, "")
                    if val != "" and fmt:
                        try:
                            val = f"{float(val):{fmt}}"
                        except (ValueError, TypeError):
                            val = str(val)
                    vals.append(str(val))
                lines.append(f"| {i} | {idea_id} | " + " | ".join(vals) + " |")
            else:
                score = result.get(primary_metric, result.get("score", "?"))
                lines.append(f"| {i} | {idea_id} | {score} |")
        lines.append("")

    # Example configs from top ideas (teaches LLM the config schema)
    if top and ideas_path.exists():
        idea_configs = parse_idea_configs(ideas_path)
        top_ids = [r.get("idea_id") for r in top[:3] if r.get("idea_id")]
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
            # Fallback: show any config
            for iid, cfg in list(idea_configs.items())[-2:]:
                lines.append(f"## Example Config ({iid})\n")
                lines.append("```yaml")
                config_yaml = yaml.dump(cfg, default_flow_style=False,
                                        sort_keys=False)
                lines.append(config_yaml.rstrip())
                lines.append("```")
                lines.append("")

    # Recent failures (help LLM avoid repeating mistakes)
    failed = {k: v for k, v in completed.items()
              if v.get("status") in ("FAILED", "ERROR")}
    if failed:
        recent_failed = sorted(failed.items(), key=lambda x: x[0])[-10:]
        lines.append("## Recent Failures (avoid these patterns)\n")
        for idea_id, metrics in recent_failed:
            error = metrics.get("error", metrics.get("status", "unknown"))
            lines.append(f"- {idea_id}: {str(error)[:100]}")
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


def append_ideas_to_md(ideas_md: list, ideas_path: Path) -> int:
    """Append formatted idea markdown strings to ideas.md. Returns count."""
    if not ideas_md:
        return 0
    with open(ideas_path, "a", encoding="utf-8") as f:
        f.write("\n")
        for md in ideas_md:
            f.write(md)
            f.write("\n")
    return len(ideas_md)


# ---------------------------------------------------------------------------
#  LLM response parsing (generic — extracts ideas from LLM output)
# ---------------------------------------------------------------------------

def parse_llm_ideas(response: str, start_id: int, cycle: int) -> list:
    """Parse LLM response into structured ideas.

    Expects the LLM to return ideas as JSON array or markdown.
    Tries JSON first, falls back to markdown parsing.

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
            idea_id = f"idea-{start_id + i:04d}"
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
        r"##\s*(?:idea-\d+:\s*)?(.+?)$\s*"
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
        idea_id = f"idea-{start_id + i:04d}"
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
    """Try to extract a JSON array from text."""
    # Find the outermost [...] block
    start = text.find("[")
    if start == -1:
        return None
    depth = 0
    end = start
    for i in range(start, len(text)):
        if text[i] == "[":
            depth += 1
        elif text[i] == "]":
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
                max_tokens: int = 8192) -> str:
    """Call Gemini API. Tries multiple models on failure."""
    models = [model, "gemini-2.5-flash", "gemini-2.5-pro"]
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
            result = _post_json(url, payload, {"Content-Type": "application/json"})
            text = (result.get("candidates", [{}])[0]
                    .get("content", {}).get("parts", [{}])[0].get("text", ""))
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
        return call_gemini(prompt, key, model=model or "gemini-2.5-flash")

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
- You receive the current leaderboard and research state
- You propose new experiments as structured ideas
- Each idea needs: a title, hypothesis, and YAML config
- The system will automatically train and evaluate your ideas
- Your ideas should build on what's working and explore new directions

## Output Format
Return a JSON array of ideas. Each idea is an object with:
- "title": short descriptive name (string)
- "hypothesis": why this might work (string)
- "config": YAML-compatible dict with experiment config (object)
- "priority": "critical" | "high" | "medium" | "low" (string, optional)
- "category": free-form label like "architecture", "hyperparameter", "loss" (string, optional)
- "parent": "none" or "idea-XXX" if building on a previous idea (string, optional)

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
    "parent": "idea-042"
  }
]
```
"""


def build_prompt(context: str, rules_content: str, num_ideas: int) -> str:
    """Build the full prompt for the LLM.

    Args:
        context: research state summary (from build_context)
        rules_content: project-specific rules (from rules_file)
        num_ideas: how many ideas to generate
    """
    parts = [DEFAULT_SYSTEM_PROMPT]

    if rules_content:
        parts.append("## Project-Specific Rules\n")
        parts.append(rules_content)
        parts.append("")

    parts.append(context)

    parts.append(f"\n## Your Task\n")
    parts.append(f"Generate exactly {num_ideas} new experiment ideas as a JSON array.")
    parts.append("Build on what's working. Avoid patterns that have failed.")
    parts.append("Each idea must have a unique, testable hypothesis.\n")

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
) -> int:
    """Run one research cycle. Returns number of ideas generated."""
    logger.info("=" * 60)
    logger.info("RESEARCH CYCLE %d (%s)", cycle, backend)
    logger.info("=" * 60)

    # 1. Build context from results
    logger.info("Step 1: Building research context...")
    context = build_context(results_dir, ideas_path, report_cfg)
    existing_ids = get_existing_idea_ids(ideas_path)
    start_id = next_idea_id(existing_ids)
    logger.info("  %d existing ideas, next ID: idea-%04d", len(existing_ids), start_id)

    # 2. Load project-specific rules if provided
    rules_content = ""
    if rules_file:
        rules_path = Path(rules_file)
        if rules_path.exists():
            rules_content = rules_path.read_text(encoding="utf-8")
            logger.info("  Loaded rules from %s (%d chars)", rules_file, len(rules_content))

    # 3. Build prompt and call LLM
    prompt = build_prompt(context, rules_content, num_ideas)
    logger.info("Step 2: Calling %s (prompt: %d chars)...", backend, len(prompt))
    response = call_llm(prompt, backend, api_key=api_key, model=model,
                        endpoint=endpoint)
    if not response:
        logger.error("LLM returned empty response — aborting cycle")
        return 0

    # 4. Parse ideas from response
    logger.info("Step 3: Parsing ideas from response...")
    ideas = parse_llm_ideas(response, start_id, cycle)
    if not ideas:
        logger.warning("Could not parse any ideas from LLM response")
        logger.debug("Response was: %s", response[:2000])
        return 0
    logger.info("  Parsed %d ideas", len(ideas))

    # 5. Format and append to ideas.md
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

    count = append_ideas_to_md(ideas_md, ideas_path)
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
    )

    sys.exit(0 if count > 0 else 1)


if __name__ == "__main__":
    main()
