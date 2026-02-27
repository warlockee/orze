"""HuggingFace model discovery — generic HTTP utility.

Queries the HuggingFace Hub API for models matching specific criteria.
No heavy dependencies (torch, transformers) — pure stdlib.

Usage:
    from orze.hf_discover import search_models
    models = search_models(pipeline_tag="image-feature-extraction", min_downloads=50000)
"""

from __future__ import annotations

import json
import logging
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

HF_API_BASE = "https://huggingface.co/api/models"
DEFAULT_CACHE_TTL = 3600 * 6  # 6 hours


def search_models(
    pipeline_tag: str = "image-feature-extraction",
    sort: str = "downloads",
    direction: str = "-1",
    limit: int = 50,
    min_downloads: int = 10000,
    filter_tags: Optional[List[str]] = None,
    cache_path: Optional[str] = None,
    cache_ttl: int = DEFAULT_CACHE_TTL,
    timeout: int = 30,
) -> List[Dict[str, Any]]:
    """Query HuggingFace API for models.

    Parameters
    ----------
    pipeline_tag : str
        HF pipeline tag (e.g. "image-feature-extraction", "image-classification").
    sort : str
        Sort field ("downloads", "likes", "lastModified").
    direction : str
        Sort direction ("-1" descending, "1" ascending).
    limit : int
        Max results to fetch (API max is 100 per page).
    min_downloads : int
        Skip models below this download count.
    filter_tags : list[str] | None
        If set, only return models whose tags intersect with this list.
    cache_path : str | None
        If set, cache results to this JSON file with TTL.
    cache_ttl : int
        Cache time-to-live in seconds.
    timeout : int
        HTTP request timeout in seconds.

    Returns
    -------
    list[dict]
        Model metadata dicts with keys: id, downloads, tags, pipeline_tag, etc.
    """
    # Check cache
    if cache_path:
        cached = _read_cache(cache_path, cache_ttl)
        if cached is not None:
            logger.debug("Using cached HF results from %s", cache_path)
            return cached

    params = {
        "pipeline_tag": pipeline_tag,
        "sort": sort,
        "direction": direction,
        "limit": str(min(limit, 100)),
    }
    url = f"{HF_API_BASE}?{urllib.parse.urlencode(params)}"
    logger.info("Querying HuggingFace API: %s", url)

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "orze/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = json.loads(resp.read().decode())
    except Exception as exc:
        logger.warning("HuggingFace API request failed: %s", exc)
        return []

    # Filter by downloads
    results = [m for m in data if m.get("downloads", 0) >= min_downloads]

    # Filter by tags
    if filter_tags:
        tag_set = set(filter_tags)
        results = [m for m in results if tag_set & set(m.get("tags", []))]

    logger.info("Found %d models (from %d raw) for pipeline_tag=%s",
                len(results), len(data), pipeline_tag)

    # Write cache
    if cache_path:
        _write_cache(cache_path, results)

    return results


def _read_cache(path: str, ttl: int) -> Optional[List[Dict[str, Any]]]:
    """Read cached results if fresh enough."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        age = time.time() - p.stat().st_mtime
        if age > ttl:
            return None
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _write_cache(path: str, data: List[Dict[str, Any]]) -> None:
    """Write results to cache file."""
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(data, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.warning("Failed to write cache to %s: %s", path, exc)


def get_model_info(model_id: str, timeout: int = 15) -> Optional[Dict[str, Any]]:
    """Fetch detailed info for a single model (config, params, gating, etc.).

    Returns dict with keys: config, transformersInfo, safetensors, gated, tags, etc.
    Returns None on error.
    """
    url = f"{HF_API_BASE}/{model_id}"
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "orze/1.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode())
    except Exception as exc:
        logger.warning("Failed to fetch model info for %s: %s", model_id, exc)
        return None
