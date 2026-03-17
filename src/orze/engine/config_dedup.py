"""Config deduplication for Orze experiment ideas.

Calling spec
------------
    from orze.engine.config_dedup import hash_config, load_hashes, save_hash, rebuild_hashes

    h = hash_config(idea_config)          # -> str  (12-char hex digest)
    cache = load_hashes(results_dir)      # -> dict[str, str]  {hash: idea_id}
    save_hash(results_dir, idea_id, cfg)  # persists one entry
    rebuild_hashes(results_dir)           # rebuilds cache from completed results

All functions are pure (no class state). The only side-effect is file I/O
on ``results_dir / "_config_hashes.json"``.
"""

import hashlib
import json
import logging
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

CACHE_FILENAME = "_config_hashes.json"

# Keys excluded from hashing — metadata only, not experiment config.
_META_KEYS = frozenset({"parent", "Parent", "category", "hypothesis", "priority", "title"})


def hash_config(config: dict) -> str:
    """Return a 12-char hex digest of an idea's config overrides."""
    clean = {
        k: v for k, v in config.items()
        if not k.startswith("_") and k not in _META_KEYS
    }
    return hashlib.md5(
        json.dumps(clean, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def load_hashes(results_dir: Path) -> dict:
    """Load the config hash -> idea_id mapping from the cache file."""
    cache_file = results_dir / CACHE_FILENAME
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_hash(results_dir: Path, idea_id: str, config: dict) -> None:
    """Persist a completed idea's config hash to the cache file."""
    cache_file = results_dir / CACHE_FILENAME
    hashes = load_hashes(results_dir)
    h = hash_config(config)
    hashes[h] = idea_id
    cache_file.write_text(json.dumps(hashes, indent=2), encoding="utf-8")


def rebuild_hashes(results_dir: Path) -> None:
    """Rebuild config hash cache from existing completed ideas' resolved configs."""
    hashes = {}
    if not results_dir.exists():
        return
    for idea_dir in results_dir.iterdir():
        if not idea_dir.is_dir() or not idea_dir.name.startswith("idea-"):
            continue
        metrics_path = idea_dir / "metrics.json"
        resolved_path = idea_dir / "resolved_config.yaml"
        if not metrics_path.exists() or not resolved_path.exists():
            continue
        try:
            m = json.loads(metrics_path.read_text(encoding="utf-8"))
            if m.get("status") != "COMPLETED":
                continue
            cfg = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
            h = hash_config(cfg)
            hashes[h] = idea_dir.name
        except Exception:
            continue
    cache_file = results_dir / CACHE_FILENAME
    cache_file.write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    logger.info("Rebuilt config hash cache: %d completed ideas", len(hashes))
