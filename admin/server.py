#!/usr/bin/env python3
"""Orze Admin Panel — FastAPI backend.

Reads filesystem state written by farm.py and exposes it via REST API.
Serves the SPA frontend from orze/admin/ui/dist/.

Usage:
    python -m orze.admin.server                     # defaults
    python -m orze.admin.server -c orze.yaml        # with project config
    python -m orze.admin.server --port 8080         # custom port
"""

import argparse
import datetime
import json
import logging
import os
import re
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from orze.farm import (
    atomic_write,
    load_project_config,
    parse_ideas,
)
try:
    from scripts.dashcam_risk.backbone_registry import BACKBONE_REGISTRY
except ImportError:
    BACKBONE_REGISTRY = {}
from orze.research_agent import append_ideas_to_md, format_idea_markdown

logger = logging.getLogger("orze.admin")

# Sensitive keys stripped from /api/config responses
_SENSITIVE_KEYS = {
    "bot_token", "webhook_url", "password", "api_key",
    "secret", "token", "GEMINI_API_KEY", "OPENAI_API_KEY",
    "ANTHROPIC_API_KEY", "LLM_API_KEY",
}

app = FastAPI(title="Orze Admin Panel", version="1.10.0")

from starlette.middleware.gzip import GZipMiddleware
app.add_middleware(GZipMiddleware, minimum_size=500)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.middleware("http")
async def cache_static_assets(request: Request, call_next):
    response = await call_next(request)
    # Hashed asset filenames are immutable — cache forever
    if "/assets/" in request.url.path:
        response.headers["Cache-Control"] = "public, max-age=31536000, immutable"
    return response

# Populated by run_admin() before uvicorn starts
_cfg: Dict[str, Any] = {}


def _results_dir() -> Path:
    return Path(_cfg.get("results_dir", "results"))


def _ideas_file() -> str:
    return _cfg.get("ideas_file", "ideas.md")



# ---------------------------------------------------------------------------
#  Backbone → HuggingFace mapping
# ---------------------------------------------------------------------------

# Well-known HF model IDs for static registry entries without explicit repo
_HF_KNOWN = {
    "dinov2_vits14": "facebook/dinov2-small",
    "dinov2_vitb14": "facebook/dinov2-base",
    "dinov2_vitl14": "facebook/dinov2-large",
    "dinov2_vitg14": "facebook/dinov2-giant",
    "convnext_tiny": "facebook/convnext-tiny-224",
    "convnext_small": "facebook/convnext-small-224",
    "dinov3_vits16": "facebook/dinov3-vits16-pretrain-lvd1689m",
    "dinov3_vitb16": "facebook/dinov3-vitb16-pretrain-lvd1689m",
    "dinov3_vitl16": "facebook/dinov3-vitl16-pretrain-lvd1689m",
    "siglip2_vit_b16": "google/siglip2-base-patch16-224",
    "siglip2_so400m": "google/siglip2-so400m-patch14-224",
    "vjepa2_vitl": "facebook/vjepa2-vitl-fpc16-256-ssv2",
    "c_radio_v4_so400m": "nvidia/C-RADIOv4-SO400M",
    "videomae_v2_vitb": "MCG-NJU/videomae-v2-base",
    "convnextv2_large": "facebook/convnextv2-large-22k-224",
}


def _get_hf_info(backbone_name: str) -> Optional[dict]:
    """Look up HuggingFace model info for a backbone name."""
    reg = BACKBONE_REGISTRY.get(backbone_name)
    if not reg:
        return None

    feature_dim = reg.get("feature_dim")
    img_size = reg.get("img_size")
    source = reg.get("source")

    # Determine HuggingFace model ID
    model_id = None
    if source == "hf_dynamic":
        model_id = reg.get("hf_model_id")
    elif source == "timm":
        model_name = reg.get("model_name", "")
        model_id = f"timm/{model_name}" if model_name else None
    elif source == "hub":
        model_id = _HF_KNOWN.get(backbone_name) or reg.get("repo")
    else:
        model_id = _HF_KNOWN.get(backbone_name)

    if not model_id:
        return None

    return {
        "model_id": model_id,
        "url": f"https://huggingface.co/{model_id}",
        "feature_dim": feature_dim,
        "img_size": img_size,
        "source": source or "known",
    }


# ---------------------------------------------------------------------------
#  TTL Cache
# ---------------------------------------------------------------------------

_cache: Dict[str, tuple] = {}


def _cached(key: str, ttl: float, fn):
    """Return cached value if within TTL, else call fn() and cache result."""
    now = time.monotonic()
    entry = _cache.get(key)
    if entry and now - entry[1] < ttl:
        return entry[0]
    val = fn()
    _cache[key] = (val, time.monotonic())
    return val


# ---------------------------------------------------------------------------
#  Helpers
# ---------------------------------------------------------------------------

def _read_json(path: Path) -> Optional[dict]:
    """Read and parse a JSON file. Returns None on any error."""
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def _strip_sensitive(obj: Any) -> Any:
    """Recursively strip sensitive keys from dicts."""
    if isinstance(obj, dict):
        return {
            k: ("***" if k.lower() in {s.lower() for s in _SENSITIVE_KEYS} else _strip_sensitive(v))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_strip_sensitive(item) for item in obj]
    return obj



def _tail_lines(path: Path, n: int = 200) -> str:
    """Read last n lines of a text file."""
    try:
        with open(path, "rb") as f:
            # Seek from end to find enough lines
            f.seek(0, 2)
            size = f.tell()
            # Read up to 1MB from the end
            read_size = min(size, 1024 * 1024)
            f.seek(max(0, size - read_size))
            data = f.read().decode("utf-8", errors="replace")
        lines = data.splitlines()
        return "\n".join(lines[-n:])
    except (OSError, ValueError):
        return ""


# ---------------------------------------------------------------------------
#  Routes
# ---------------------------------------------------------------------------

@app.get("/api/status")
async def get_status():
    """Read results/status.json and return as-is."""
    path = _results_dir() / "status.json"
    data = _read_json(path)
    if data is None:
        raise HTTPException(404, "status.json not found or unreadable")
    return data


def _read_admin_cache() -> Optional[dict]:
    """Read the pre-aggregated admin cache written by farm.py."""
    return _read_json(_results_dir() / "_admin_cache.json")


@app.get("/api/fleet")
async def get_fleet():
    """Host heartbeats + GPU details from admin cache."""
    def _get():
        cache = _read_admin_cache()
        if cache and cache.get("fleet"):
            return cache["fleet"]
        return {"heartbeats": [], "local_gpus": []}
    return _cached("fleet", 5.0, _get)


@app.get("/api/runs")
async def get_runs(limit: int = 50):
    """Active runs from status.json + recent completed/failed from filesystem.

    Only scans recent idea dirs (by mtime) to avoid slow scans of 20k+ dirs.
    """
    status_data = _read_json(_results_dir() / "status.json")
    raw_active = status_data.get("active", []) if status_data else []

    # Enrich active runs with human-readable titles from ideas.md
    ideas = _cached("parsed_ideas", 10.0, lambda: parse_ideas(_ideas_file()))
    active = []
    for r in raw_active:
        idea_id = r.get("idea_id", "")
        base_id = idea_id.split("~")[0]  # idea-1234~sweep_param -> idea-1234
        title = ideas.get(base_id, {}).get("title", "") if ideas else ""
        active.append({**r, "title": title})

    # Also include top_results from status.json as completed runs
    top_results = status_data.get("top_results", []) if status_data else []

    # Get recent completed/failed: sort idea dirs by mtime (newest first), stop early
    recent = []
    rd = _results_dir()
    if rd.exists():
        entries = []
        try:
            with os.scandir(rd) as it:
                for entry in it:
                    if entry.is_dir() and entry.name.startswith("idea-"):
                        try:
                            entries.append((entry.stat().st_mtime, entry.path, entry.name))
                        except OSError:
                            pass
        except OSError:
            pass
        # Sort by mtime descending, take only recent
        entries.sort(key=lambda x: x[0], reverse=True)
        for _, path, name in entries[:limit]:
            metrics = _read_json(Path(path) / "metrics.json")
            if metrics is None:
                continue
            metrics["idea_id"] = name
            claim_data = _read_json(Path(path) / "claim.json")
            if claim_data:
                metrics["claimed_by"] = claim_data.get("host")
                metrics["claimed_gpu"] = claim_data.get("gpu")
            recent.append(metrics)

    return {
        "active": active,
        "top_results": top_results,
        "recent": recent,
    }


@app.get("/api/run/detail")
async def get_run(idea_id: str):
    """Read metrics.json + claim.json for a specific idea."""
    if not re.match(r"^idea-\d+", idea_id):
        raise HTTPException(400, "Invalid idea_id format")

    idea_dir = _results_dir() / idea_id
    if not idea_dir.is_dir():
        raise HTTPException(404, f"{idea_id} not found")

    metrics = _read_json(idea_dir / "metrics.json")
    claim_data = _read_json(idea_dir / "claim.json")

    return {
        "idea_id": idea_id,
        "metrics": metrics,
        "claim": claim_data,
    }


@app.get("/api/run/log")
async def get_run_log(idea_id: str):
    """Tail training.log (last 200 lines)."""
    if not re.match(r"^idea-\d+", idea_id):
        raise HTTPException(400, "Invalid idea_id format")

    idea_dir = _results_dir() / idea_id
    # Try common log file names
    log_path = None
    for name in ("train_output.log", "training.log", "output.log"):
        p = idea_dir / name
        if p.exists():
            log_path = p
            break
    if log_path is None:
        raise HTTPException(404, f"No log file found for {idea_id}")

    text = _tail_lines(log_path, n=200)
    return {"idea_id": idea_id, "log": text}


@app.get("/api/leaderboard")
async def get_leaderboard():
    """Top models — reads _leaderboard.json written by farm.py update_report()."""
    def _read_lb():
        lb = _read_json(_results_dir() / "_leaderboard.json")
        if lb and lb.get("top"):
            return lb
        metric = (_cfg.get("report") or {}).get("primary_metric", "test_accuracy")
        return {"top": [], "metric": metric}
    return _cached("leaderboard", 10, _read_lb)


@app.get("/api/ideas")
async def get_ideas():
    """Parse ideas.md and return all ideas."""
    ideas = parse_ideas(_ideas_file())
    return {"ideas": ideas, "count": len(ideas)}


@app.get("/api/queue")
async def get_queue(
    page: int = 1,
    per_page: int = 50,
    status_filter: str = "",
    search: str = "",
):
    """Return paginated, filterable queue from admin cache."""
    cache = _cached("admin_cache", 5.0,
                    lambda: _read_admin_cache() or {})
    queue_data = cache.get("queue", {})
    all_items = queue_data.get("items", [])
    all_statuses = queue_data.get("counts", {})

    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
    status_order = {"pending": 0, "running": 1, "completed": 2, "failed": 3, "error": 4}

    # Enrich with HF info and filter
    results = []
    for item in all_items:
        if status_filter and item.get("status") != status_filter:
            continue
        if search:
            s = search.lower()
            if (s not in item.get("idea_id", "").lower()
                    and s not in item.get("title", "").lower()):
                continue
        # Add HF info on-demand (cheap lookup, no I/O)
        item["huggingface"] = _get_hf_info(
            item.get("config", {}).get("backbone", {}).get("name", ""))
        results.append(item)

    results.sort(key=lambda r: (
        status_order.get(r.get("status", ""), 9),
        priority_order.get(r.get("priority", ""), 9),
        r.get("idea_id", ""),
    ))

    total_filtered = len(results)
    start = (page - 1) * per_page
    page_items = results[start:start + per_page]
    total_pages = max(1, (total_filtered + per_page - 1) // per_page)

    return {
        "queue": page_items,
        "total": total_filtered,
        "total_all": queue_data.get("total_all", 0),
        "page": page,
        "per_page": per_page,
        "total_pages": total_pages,
        "counts": all_statuses,
    }


@app.get("/api/alerts")
async def get_alerts():
    """Alerts from admin cache."""
    def _get():
        cache = _read_admin_cache()
        if cache and cache.get("alerts"):
            return cache["alerts"]
        return {"alerts": [], "count": 0}
    return _cached("alerts", 5.0, _get)


@app.get("/api/config")
async def get_config():
    """Read orze.yaml, strip tokens/passwords."""
    # Return the loaded config with sensitive values masked
    return _strip_sensitive(_cfg)


@app.post("/api/ideas")
async def post_idea(request: Request):
    """Append a new idea to ideas.md."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    title = body.get("title")
    hypothesis = body.get("hypothesis", "")
    config = body.get("config", {})
    priority = body.get("priority", "high")
    category = body.get("category", "architecture")
    parent = body.get("parent", "none")

    if not title:
        raise HTTPException(400, "title is required")
    if not isinstance(config, dict):
        raise HTTPException(400, "config must be a dict")

    # Determine next idea ID
    ideas_path = Path(_ideas_file())
    existing = parse_ideas(_ideas_file())
    if existing:
        max_num = max(
            int(k.replace("idea-", "")) for k in existing.keys()
        )
    else:
        max_num = 0
    next_num = max_num + 1
    idea_id = f"idea-{next_num:04d}"

    md = format_idea_markdown(
        idea_id=idea_id,
        title=title,
        hypothesis=hypothesis or title,
        config=config,
        priority=priority,
        category=category,
        parent=parent,
    )
    count = append_ideas_to_md([md], ideas_path)

    return {"idea_id": idea_id, "appended": count}


@app.post("/api/actions/stop")
async def action_stop():
    """Write results/.orze_stop_all sentinel to stop all farm instances."""
    stop_path = _results_dir() / ".orze_stop_all"
    atomic_write(stop_path, datetime.datetime.now().isoformat())
    return {"status": "ok", "message": "Stop sentinel written", "path": str(stop_path)}


@app.post("/api/actions/kill")
async def action_kill(request: Request):
    """Write results/idea-{id}/.kill sentinel to kill a specific run."""
    try:
        body = await request.json()
    except Exception:
        raise HTTPException(400, "Invalid JSON body")

    idea_id = body.get("idea_id")
    if not idea_id or not re.match(r"^idea-\d+", idea_id):
        raise HTTPException(400, "Valid idea_id required (e.g. idea-0042)")

    idea_dir = _results_dir() / idea_id
    idea_dir.mkdir(parents=True, exist_ok=True)
    kill_path = idea_dir / ".kill"
    atomic_write(kill_path, datetime.datetime.now().isoformat())

    return {"status": "ok", "message": f"Kill sentinel written for {idea_id}", "path": str(kill_path)}


# ---------------------------------------------------------------------------
#  Static files (SPA frontend)
# ---------------------------------------------------------------------------

_ui_dist = Path(__file__).parent / "ui" / "dist"
if _ui_dist.is_dir():
    app.mount("/", StaticFiles(directory=str(_ui_dist), html=True), name="spa")


# ---------------------------------------------------------------------------
#  Entrypoint
# ---------------------------------------------------------------------------

def run_admin(cfg: dict, host: str = "0.0.0.0", port: int = 8787):
    """Start the admin panel server."""
    global _cfg
    _cfg = cfg

    import uvicorn

    logger.info("Starting Orze Admin Panel on %s:%d", host, port)
    logger.info("  results_dir: %s", cfg.get("results_dir", "results"))
    logger.info("  ideas_file:  %s", cfg.get("ideas_file", "ideas.md"))

    # Pre-warm: read admin cache once so first requests are instant
    try:
        ac = _read_admin_cache()
        if ac:
            _cache["admin_cache"] = (ac, time.monotonic())
            _cache["fleet"] = (ac.get("fleet", {"heartbeats": [], "local_gpus": []}), time.monotonic())
            _cache["alerts"] = (ac.get("alerts", {"alerts": [], "count": 0}), time.monotonic())
        lb = _read_json(_results_dir() / "_leaderboard.json")
        if lb:
            _cache["leaderboard"] = (lb, time.monotonic())
        logger.info("Cache pre-warmed")
    except Exception as e:
        logger.warning("Cache pre-warm failed: %s", e)

    uvicorn.run(app, host=host, port=port, log_level="info")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    parser = argparse.ArgumentParser(description="Orze Admin Panel")
    parser.add_argument("-c", "--config", default="orze.yaml",
                        help="Path to orze.yaml")
    parser.add_argument("--host", default="0.0.0.0",
                        help="Bind host (default: 0.0.0.0)")
    parser.add_argument("--port", type=int, default=8787,
                        help="Bind port (default: 8787)")
    args = parser.parse_args()

    cfg = load_project_config(args.config)
    run_admin(cfg, host=args.host, port=args.port)
