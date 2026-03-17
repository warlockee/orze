"""Parsing and sweep expansion of ideas.md files.

CALLING SPEC:
    parse_ideas(path: str) -> Dict[str, dict]
        Parse an ideas.md file into {idea_id: {title, priority, config, raw}}.
        Cached by file mtime — safe to call repeatedly. Returns {} on error.

    expand_sweeps(ideas: Dict[str, dict], max_combos: int = 20) -> Dict[str, dict]
        Expand ideas with list-valued hyperparams into Cartesian-product
        sub-runs. Sub-run IDs use format idea-XXX-ht-N. Ideas already
        containing '~' or exceeding max_combos pass through unchanged.
"""
import re
import yaml
import logging
import copy
from typing import Dict, Any
from orze.core.config import _sanitize_config
from pathlib import Path

_SWEEP_BLOCKLIST = frozenset({
    "betas", "split_ratio", "stack_layers", "stack_dims",
    "downsampling_factors", "num_heads", "feedforward_dim",
    "output_range", "augmentations", "transforms",
})
logger = logging.getLogger("orze")

_parse_ideas_cache: Dict[str, Any] = {"mtime": 0.0, "result": {}, "path": ""}


def parse_ideas(path: str) -> Dict[str, dict]:
    """Parse ideas.md into {idea_id: {title, priority, config, raw}}.
    Results are cached by file mtime to avoid re-parsing on every iteration.
    """
    try:
        p = Path(path)
        mtime = p.stat().st_mtime
        if (mtime == _parse_ideas_cache["mtime"]
                and path == _parse_ideas_cache["path"]
                and _parse_ideas_cache["result"]):
            return _parse_ideas_cache["result"]
        text = p.read_text(encoding="utf-8")
    except OSError:
        return {}
    ideas = {}
    pattern = re.compile(r"^## (idea-[a-z0-9]+):\s*(.+?)$", re.MULTILINE)
    matches = list(pattern.finditer(text))

    for i, m in enumerate(matches):
        idea_id = m.group(1)
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw = text[start:end]

        pri_match = re.search(r"\*\*Priority\*\*:\s*(\w+)", raw)
        priority = pri_match.group(1).lower() if pri_match else "medium"

        yaml_match = re.search(r"```ya?ml\s*\n(.*?)```", raw, re.DOTALL)
        config = {}
        if yaml_match:
            try:
                config = yaml.safe_load(yaml_match.group(1)) or {}
            except yaml.YAMLError:
                pass

        # Sanitize config: replace non-numeric values in numeric fields
        config = _sanitize_config(config)

        ideas[idea_id] = {
            "title": title,
            "priority": priority,
            "config": config,
            "raw": raw.strip(),
        }
    _parse_ideas_cache["mtime"] = mtime
    _parse_ideas_cache["path"] = path
    _parse_ideas_cache["result"] = ideas
    return ideas


def _find_sweep_keys(config: dict, prefix: str = "") -> Dict[str, list]:
    """Recursively find list-valued scalar keys suitable for sweeping.
    Returns dict of dotpath -> list-of-values.
    """
    found = {}
    for key, val in config.items():
        dotpath = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            found.update(_find_sweep_keys(val, dotpath))
        elif isinstance(val, list) and key not in _SWEEP_BLOCKLIST:
            # Only sweep if all elements are scalars (int, float, str, bool)
            if val and all(isinstance(v, (int, float, str, bool)) for v in val):
                found[dotpath] = val
    return found


def _set_nested(config: dict, dotpath: str, value):
    """Set a value at a dot-separated path in a nested dict."""
    keys = dotpath.split(".")
    d = config
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def expand_sweeps(ideas: Dict[str, dict],
                  max_combos: int = 20) -> Dict[str, dict]:
    """Expand ideas with list-valued HPs into sub-run entries.

    Sub-run IDs use format: idea-123~key=val~key2=val2
    Original ideas with no sweepable keys pass through unchanged.
    Ideas whose ID already contains '~' are skipped (no double-expansion).
    """
    import itertools

    expanded = {}
    for idea_id, idea in ideas.items():
        # Skip already-expanded sub-runs
        if "~" in idea_id:
            expanded[idea_id] = idea
            continue

        sweep_keys = _find_sweep_keys(idea.get("config", {}))
        if not sweep_keys:
            expanded[idea_id] = idea
            continue

        # Compute Cartesian product
        keys = sorted(sweep_keys.keys())
        values = [sweep_keys[k] for k in keys]
        combos = list(itertools.product(*values))

        if len(combos) > max_combos:
            logger.warning(
                "Sweep for %s has %d combos (max %d), skipping expansion",
                idea_id, len(combos), max_combos)
            expanded[idea_id] = idea
            continue

        logger.info("Expanding %s into %d sweep sub-runs (%s)",
                    idea_id, len(combos),
                    ", ".join(f"{k}={len(v)}" for k, v in zip(keys, values)))

        for i, combo in enumerate(combos, 1):
            sub_config = copy.deepcopy(idea["config"])
            for k, v in zip(keys, combo):
                _set_nested(sub_config, k, v)

            # Use requested -ht-N naming convention
            sub_id = f"{idea_id}-ht-{i}"
            expanded[sub_id] = {
                "title": idea["title"],
                "priority": idea.get("priority", "medium"),
                "config": sub_config,
                "raw": idea.get("raw", ""),
                "_sweep_parent": idea_id,
            }

    return expanded


