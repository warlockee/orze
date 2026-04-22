"""Merged integrity module — sealed hashes, guardrails, config dedup, stale DB relocation.

Consolidates four previously-separate modules into one integrity surface:

    * Sealed file manifest (compute_sealed_hashes, write/load manifest,
      verify_sealed_files, validate_metrics) — was engine/sealed.py.
    * Runtime guardrails (check_base_config_drift, check_identical_results,
      validate_avg_metric) — was engine/guardrails.py.
    * Config deduplication (hash_config, load_hashes, save_hash,
      rebuild_hashes, CACHE_FILENAME) — was engine/config_dedup.py.
    * Stale zero-byte DB relocation (relocate_zero_byte_dbs,
      CANONICAL_DB_NAMES) — was engine/stale_dbs.py.

All public names remain importable from the old paths via thin
shim modules (engine/sealed.py, engine/guardrails.py,
engine/config_dedup.py, engine/stale_dbs.py) which simply re-export.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import shutil
import time
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import yaml

logger = logging.getLogger("orze")


# ---------------------------------------------------------------------------
# Sealed files (was engine/sealed.py)
# ---------------------------------------------------------------------------

_MANIFEST_FILE = ".sealed_hashes"


def compute_sealed_hashes(file_list: List[str]) -> Dict[str, str]:
    hashes = {}
    for fpath in file_list:
        p = Path(fpath)
        if not p.exists():
            logger.warning("Sealed file does not exist: %s", fpath)
            continue
        try:
            hashes[fpath] = hashlib.sha256(p.read_bytes()).hexdigest()
        except OSError as e:
            logger.warning("Could not hash sealed file %s: %s", fpath, e)
    return hashes


def write_sealed_manifest(results_dir: Path, hashes: Dict[str, str]) -> None:
    manifest_path = Path(results_dir) / _MANIFEST_FILE
    try:
        manifest_path.write_text(json.dumps(hashes, indent=2), encoding="utf-8")
        logger.info("Sealed manifest written: %d files tracked", len(hashes))
    except OSError as e:
        logger.error("Could not write sealed manifest: %s", e)


def load_sealed_manifest(results_dir: Path) -> Dict[str, str]:
    manifest_path = Path(results_dir) / _MANIFEST_FILE
    if not manifest_path.exists():
        return {}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return {}


def verify_sealed_files(file_list: List[str],
                        manifest: Dict[str, str]) -> List[str]:
    if not manifest:
        return []
    changed = []
    for fpath in file_list:
        expected = manifest.get(fpath)
        if expected is None:
            continue
        p = Path(fpath)
        if not p.exists():
            changed.append(fpath)
            continue
        try:
            actual = hashlib.sha256(p.read_bytes()).hexdigest()
            if actual != expected:
                changed.append(fpath)
        except OSError:
            changed.append(fpath)
    return changed


def validate_metrics(metrics: dict, cfg: dict) -> Tuple[bool, str]:
    if not metrics or metrics.get("status") != "COMPLETED":
        return True, ""
    validation_cfg = cfg.get("metric_validation", {})
    reject_nan = validation_cfg.get("reject_nan", True)
    reject_inf = validation_cfg.get("reject_inf", True)
    min_values = validation_cfg.get("min_value", {})
    max_values = validation_cfg.get("max_value", {})
    for key, val in metrics.items():
        if key in ("status", "timestamp", "error", "idea_id"):
            continue
        if not isinstance(val, (int, float)):
            continue
        if reject_nan and math.isnan(val):
            return False, f"Metric '{key}' is NaN"
        if reject_inf and math.isinf(val):
            return False, f"Metric '{key}' is inf"
        if key in min_values and val < min_values[key]:
            return False, f"Metric '{key}'={val} below minimum {min_values[key]}"
        if key in max_values and val > max_values[key]:
            return False, f"Metric '{key}'={val} above maximum {max_values[key]}"
    return True, ""


# ---------------------------------------------------------------------------
# Runtime guardrails (was engine/guardrails.py)
# ---------------------------------------------------------------------------

_CONFIG_HASH_FILE = ".base_config_hash"


def check_base_config_drift(results_dir: Path,
                            base_config_path: str) -> Optional[str]:
    hash_file = Path(results_dir) / _CONFIG_HASH_FILE
    try:
        current_hash = hashlib.sha256(
            Path(base_config_path).read_bytes()).hexdigest()[:16]
    except OSError:
        return None
    old_hash = None
    if hash_file.exists():
        try:
            old_hash = hash_file.read_text().strip()
        except OSError:
            pass
    try:
        hash_file.write_text(current_hash)
    except OSError:
        pass
    if old_hash and old_hash != current_hash:
        return (
            f"base_config changed since last run ({old_hash[:8]}→{current_hash[:8]}). "
            f"Old results may not be comparable. Consider `orze reset --all`."
        )
    return None


def check_identical_results(recent_metrics: List[Dict],
                            primary_metric: str,
                            threshold: int = 3) -> Optional[str]:
    if len(recent_metrics) < threshold:
        return None
    value_groups: Dict[str, list] = {}
    for item in recent_metrics:
        val = item.get("metrics", {}).get(primary_metric)
        if val is None:
            continue
        key = f"{val:.4f}"
        value_groups.setdefault(key, []).append(item.get("idea_id", "?"))
    for val, ids in value_groups.items():
        if len(ids) >= threshold:
            return (
                f"ANOMALY: {len(ids)} experiments have identical "
                f"{primary_metric}={val}: {ids[:5]}. "
                f"Config overrides may not be reaching the train script."
            )
    return None


def validate_avg_metric(metrics: dict,
                        primary_metric: str,
                        per_prefix: str = "",
                        tolerance: float = 0.5) -> Optional[str]:
    if primary_metric not in metrics:
        return None
    if not per_prefix:
        for strip in ("avg_", "mean_", "average_"):
            if primary_metric.startswith(strip):
                per_prefix = primary_metric[len(strip):] + "_"
                break
    if not per_prefix:
        return None
    per_vals = {k: v for k, v in metrics.items()
                if k.startswith(per_prefix) and isinstance(v, (int, float))}
    if len(per_vals) < 2:
        return None
    expected = sum(per_vals.values()) / len(per_vals)
    actual = metrics[primary_metric]
    if abs(expected - actual) > tolerance:
        return (f"{primary_metric}={actual:.2f} but mean of "
                f"{len(per_vals)} sub-metrics = {expected:.2f}")
    return None


# ---------------------------------------------------------------------------
# Config deduplication (was engine/config_dedup.py)
# ---------------------------------------------------------------------------

CACHE_FILENAME = "_config_hashes.json"
_META_KEYS = frozenset({"parent", "Parent", "category", "hypothesis",
                        "priority", "title"})


def hash_config(config: dict) -> str:
    clean = {k: v for k, v in config.items()
             if not k.startswith("_") and k not in _META_KEYS}
    return hashlib.md5(
        json.dumps(clean, sort_keys=True, default=str).encode()
    ).hexdigest()[:12]


def load_hashes(results_dir: Path, cfg: dict = None) -> dict:
    """Load config hash cache from .orze/state/config_hashes.json."""
    if cfg:
        from orze.core.config import orze_path
        cache_file = orze_path(cfg, "state", "config_hashes.json")
    else:
        # Fallback for legacy calls without cfg
        cache_file = Path(results_dir) / CACHE_FILENAME
    if cache_file.exists():
        try:
            return json.loads(cache_file.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def save_hash(results_dir: Path, idea_id: str, config: dict, cfg: dict = None) -> None:
    """Save config hash to .orze/state/config_hashes.json."""
    if cfg:
        from orze.core.config import orze_path
        cache_file = orze_path(cfg, "state", "config_hashes.json")
        hashes = load_hashes(results_dir, cfg)
    else:
        # Fallback for legacy calls
        cache_file = Path(results_dir) / CACHE_FILENAME
        hashes = load_hashes(results_dir)
    hashes[hash_config(config)] = idea_id
    cache_file.write_text(json.dumps(hashes, indent=2), encoding="utf-8")


def rebuild_hashes(results_dir: Path, cfg: dict = None) -> None:
    """Rebuild config hash cache from completed ideas."""
    results_dir = Path(results_dir)
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
            cfg_data = yaml.safe_load(resolved_path.read_text(encoding="utf-8")) or {}
            hashes[hash_config(cfg_data)] = idea_dir.name
        except Exception:
            continue
    if cfg:
        from orze.core.config import orze_path
        cache_file = orze_path(cfg, "state", "config_hashes.json")
    else:
        cache_file = results_dir / CACHE_FILENAME
    cache_file.write_text(json.dumps(hashes, indent=2), encoding="utf-8")
    logger.info("Rebuilt config hash cache: %d completed ideas", len(hashes))


# ---------------------------------------------------------------------------
# Stale DB relocation (was engine/stale_dbs.py)
# ---------------------------------------------------------------------------

CANONICAL_DB_NAMES = (
    "idea_lake.db", "queue.db", "orze.db",
    "lake.db", "orze_queue.db", "ideas.db",
)


def relocate_zero_byte_dbs(cwd: Path,
                           stale_dir: Path,
                           names: Iterable[str] = CANONICAL_DB_NAMES
                           ) -> List[Tuple[Path, Path]]:
    moved: List[Tuple[Path, Path]] = []
    for name in names:
        p = Path(cwd) / name
        if not p.exists() or p.is_dir():
            continue
        try:
            if p.stat().st_size != 0:
                continue
        except OSError:
            continue
        try:
            Path(stale_dir).mkdir(parents=True, exist_ok=True)
            dest = Path(stale_dir) / f"{name}.{int(time.time())}"
            shutil.move(str(p), str(dest))
            moved.append((p, dest))
        except OSError as e:
            logger.warning("Could not move stale %s: %s", p, e)
    return moved
