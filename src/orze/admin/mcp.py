"""MCP (Model Context Protocol) HTTP handler for Orze.

CALLING SPEC:
    mount_mcp(app: FastAPI, cfg_getter: Callable) -> None
        Mounts MCP endpoint at /mcp on a FastAPI app.
        cfg_getter returns the orze config dict.

    Implements MCP Streamable HTTP transport (JSON-RPC over POST).
    Claude Code connects via .mcp.json: {"orze": {"type": "http", "url": "http://localhost:8787/mcp"}}

    Tools exposed:
        orze_status()                        -> pipeline status (GPUs, queue, active runs)
        orze_leaderboard(top_n=10)           -> top models ranked by primary metric
        orze_queue(limit=20)                 -> pending ideas in queue
        orze_run_detail(idea_id)             -> metrics + config for one run
        orze_run_log(idea_id, tail=100)      -> tail of training log
        orze_add_idea(title, config_yaml, hypothesis?, priority?, category?, parent?)
        orze_nodes()                         -> cluster node status
"""

import json
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from orze.core.config import orze_path

logger = logging.getLogger("orze.mcp")

# MCP protocol version
PROTOCOL_VERSION = "2025-03-26"

# --- Tool definitions (JSON Schema for each tool's input) ---

TOOLS = [
    {
        "name": "orze_status",
        "description": "Get current Orze pipeline status: active runs, free GPUs, queue depth, completed/failed counts, top results.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
    {
        "name": "orze_leaderboard",
        "description": "Get top models ranked by primary metric.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "top_n": {"type": "integer", "description": "Number of results (default 10)", "default": 10},
            },
        },
    },
    {
        "name": "orze_queue",
        "description": "Get pending ideas waiting to be trained.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "limit": {"type": "integer", "description": "Max results (default 20)", "default": 20},
            },
        },
    },
    {
        "name": "orze_run_detail",
        "description": "Get metrics, config, and status for a specific experiment run.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "idea_id": {"type": "string", "description": "Experiment ID (e.g. 'idea-001')"},
            },
            "required": ["idea_id"],
        },
    },
    {
        "name": "orze_run_log",
        "description": "Get the tail of a training log for a specific run.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "idea_id": {"type": "string", "description": "Experiment ID"},
                "tail": {"type": "integer", "description": "Number of lines from end (default 100)", "default": 100},
            },
            "required": ["idea_id"],
        },
    },
    {
        "name": "orze_add_idea",
        "description": "Add a new experiment idea to the queue. Appends to ideas.md.",
        "inputSchema": {
            "type": "object",
            "properties": {
                "title": {"type": "string", "description": "Short title for the experiment"},
                "config_yaml": {"type": "string", "description": "YAML config block (model, training params, etc.)"},
                "hypothesis": {"type": "string", "description": "Why this might work"},
                "priority": {"type": "string", "enum": ["low", "medium", "high"], "default": "medium"},
                "category": {"type": "string", "description": "Category (architecture, hyperparameter, data, etc.)"},
                "parent": {"type": "string", "description": "Parent idea ID (e.g. 'idea-001')"},
            },
            "required": ["title", "config_yaml"],
        },
    },
    {
        "name": "orze_nodes",
        "description": "Get cluster node status: host, GPUs busy/total, utilization per node.",
        "inputSchema": {
            "type": "object",
            "properties": {},
        },
    },
]


# --- Tool implementations ---

def _read_json(path: Path) -> Optional[dict]:
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            pass
    return None


def _tool_status(args: dict, cfg: dict) -> str:
    results_dir = Path(cfg.get("results_dir", "orze_results"))
    status = _read_json(results_dir / "status.json")
    if not status:
        return "No status.json found. Is orze running?"
    # Compact summary
    return json.dumps({
        "iteration": status.get("iteration"),
        "active": status.get("active", []),
        "free_gpus": status.get("free_gpus", []),
        "queue_depth": status.get("queue_depth"),
        "completed": status.get("completed"),
        "failed": status.get("failed"),
        "top_results": status.get("top_results", [])[:5],
        "timestamp": status.get("timestamp"),
    }, indent=2)


def _tool_leaderboard(args: dict, cfg: dict) -> str:
    results_dir = Path(cfg.get("results_dir", "orze_results"))
    top_n = args.get("top_n", 10)
    lb = _read_json(results_dir / "_leaderboard.json")
    if not lb:
        return "No leaderboard data found."
    entries = (lb.get("top") or [])[:top_n]
    lines = []
    for i, e in enumerate(entries, 1):
        val = e.get("metric_value", "?")
        if isinstance(val, float):
            val = f"{val:.4f}"
        lines.append(f"{i}. {e.get('idea_id', '?')} — {e.get('title', '')[:50]} = {val}")
    return "\n".join(lines) if lines else "Leaderboard is empty."


def _tool_queue(args: dict, cfg: dict) -> str:
    results_dir = Path(cfg.get("results_dir", "orze_results"))
    limit = args.get("limit", 20)
    cache = _read_json(orze_path(cfg, "state", "admin_cache.json"))
    if cache and cache.get("queue"):
        queue_data = cache["queue"]
        # queue can be a list or a dict with 'items' key
        if isinstance(queue_data, dict):
            items = (queue_data.get("items") or [])[:limit]
        else:
            items = queue_data[:limit]
        lines = [f"- {q.get('idea_id', '?')}: {q.get('title', '')[:60]} [{q.get('priority', 'medium')}]"
                 for q in items]
        return "\n".join(lines) if lines else "Queue is empty."
    # Fallback: try idea_lake
    try:
        from orze.idea_lake import IdeaLake
        lake_path = Path(cfg.get("idea_lake_db") or Path(cfg.get("ideas_file", "ideas.md")).parent / "idea_lake.db")
        if lake_path.exists():
            lake = IdeaLake(str(lake_path))
            rows = lake.get_queue(limit=limit)
            lake.close()
            lines = [f"- {r['idea_id']}: {r['title'][:60]} [{r['priority']}]" for r in rows]
            return "\n".join(lines) if lines else "Queue is empty."
    except Exception:
        pass
    return "No queue data available."


def _tool_run_detail(args: dict, cfg: dict) -> str:
    idea_id = args.get("idea_id", "")
    if not idea_id:
        return "Error: idea_id is required."
    results_dir = Path(cfg.get("results_dir", "orze_results"))
    idea_dir = results_dir / idea_id
    if not idea_dir.is_dir():
        return f"No results found for {idea_id}."
    result = {}
    metrics = _read_json(idea_dir / "metrics.json")
    if metrics:
        result["metrics"] = metrics
    claim = _read_json(idea_dir / "claim.json")
    if claim:
        result["claim"] = claim
    eval_file = cfg.get("eval_output", "eval_report.json")
    eval_data = _read_json(idea_dir / eval_file)
    if eval_data:
        result["eval"] = eval_data
    return json.dumps(result, indent=2) if result else f"No data for {idea_id}."


def _tool_run_log(args: dict, cfg: dict) -> str:
    idea_id = args.get("idea_id", "")
    tail = args.get("tail", 100)
    if not idea_id:
        return "Error: idea_id is required."
    results_dir = Path(cfg.get("results_dir", "orze_results"))
    log_path = results_dir / idea_id / "train_output.log"
    if not log_path.exists():
        return f"No training log found for {idea_id}."
    try:
        lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
        return "\n".join(lines[-tail:])
    except Exception as e:
        return f"Error reading log: {e}"


def _tool_add_idea(args: dict, cfg: dict) -> str:
    title = args.get("title", "").strip()
    config_yaml = args.get("config_yaml", "").strip()
    if not title or not config_yaml:
        return "Error: title and config_yaml are required."
    hypothesis = args.get("hypothesis", "")
    priority = args.get("priority", "medium")
    category = args.get("category", "")
    parent = args.get("parent", "none")

    # Generate idea ID
    import hashlib
    import time
    hash_input = f"{title}{time.time()}"
    idea_id = "idea-" + hashlib.md5(hash_input.encode()).hexdigest()[:6]

    # Build markdown
    lines = [f"\n## {idea_id}: {title}"]
    lines.append(f"- **Priority**: {priority}")
    if category:
        lines.append(f"- **Category**: {category}")
    lines.append(f"- **Parent**: {parent}")
    if hypothesis:
        lines.append(f"- **Hypothesis**: {hypothesis}")
    lines.append("")
    lines.append("```yaml")
    lines.append(config_yaml)
    lines.append("```")
    lines.append("")

    ideas_file = cfg.get("ideas_file", "ideas.md")
    try:
        with open(ideas_file, "a", encoding="utf-8") as f:
            f.write("\n".join(lines))
        return f"Added {idea_id}: {title}"
    except Exception as e:
        return f"Error appending to {ideas_file}: {e}"


def _tool_nodes(args: dict, cfg: dict) -> str:
    results_dir = Path(cfg.get("results_dir", "orze_results"))
    nodes = []
    for p in sorted(results_dir.glob("_host_*.json")):
        data = _read_json(p)
        if not data or not isinstance(data, dict):
            continue
        active = data.get("active", [])
        free = data.get("free_gpus", [])
        nodes.append({
            "host": data.get("host", "?"),
            "status": data.get("status", "running"),
            "gpus_busy": len(active),
            "gpus_total": len(active) + len(free),
            "active_runs": [a.get("idea_id") for a in active],
            "version": data.get("orze_version", "?"),
        })
    return json.dumps(nodes, indent=2) if nodes else "No node heartbeats found."


# Tool dispatch
_TOOL_HANDLERS = {
    "orze_status": _tool_status,
    "orze_leaderboard": _tool_leaderboard,
    "orze_queue": _tool_queue,
    "orze_run_detail": _tool_run_detail,
    "orze_run_log": _tool_run_log,
    "orze_add_idea": _tool_add_idea,
    "orze_nodes": _tool_nodes,
}


# --- MCP JSON-RPC handler ---

def _handle_initialize(params: dict) -> dict:
    return {
        "protocolVersion": PROTOCOL_VERSION,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "orze", "version": "1.0.0"},
    }


def _handle_tools_list(params: dict) -> dict:
    return {"tools": TOOLS}


def _handle_tools_call(params: dict, cfg: dict) -> dict:
    name = params.get("name", "")
    args = params.get("arguments", {})
    handler = _TOOL_HANDLERS.get(name)
    if not handler:
        return {
            "content": [{"type": "text", "text": f"Unknown tool: {name}"}],
            "isError": True,
        }
    try:
        result = handler(args, cfg)
        return {"content": [{"type": "text", "text": result}]}
    except Exception as e:
        logger.warning("MCP tool %s error: %s", name, e)
        return {
            "content": [{"type": "text", "text": f"Error: {e}"}],
            "isError": True,
        }


def _handle_rpc(body: dict, cfg: dict) -> dict:
    """Handle a single JSON-RPC request. Returns a JSON-RPC response dict."""
    method = body.get("method", "")
    params = body.get("params", {})
    req_id = body.get("id")

    handlers = {
        "initialize": lambda p: _handle_initialize(p),
        "notifications/initialized": lambda p: None,  # notification, no response
        "tools/list": lambda p: _handle_tools_list(p),
        "tools/call": lambda p: _handle_tools_call(p, cfg),
        "ping": lambda p: {},
    }

    handler = handlers.get(method)
    if handler is None:
        return {
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {"code": -32601, "message": f"Method not found: {method}"},
        }

    result = handler(params)
    if result is None:
        # Notification — no response needed, but we return accepted
        return None

    return {
        "jsonrpc": "2.0",
        "id": req_id,
        "result": result,
    }


# --- FastAPI mount ---

def mount_mcp(app: FastAPI, cfg_getter: Callable[[], dict]) -> None:
    """Mount MCP endpoint at /mcp on a FastAPI app."""

    @app.post("/mcp")
    async def mcp_endpoint(request: Request):
        try:
            body = await request.json()
        except Exception:
            return JSONResponse(
                {"jsonrpc": "2.0", "id": None,
                 "error": {"code": -32700, "message": "Parse error"}},
                status_code=400,
            )

        cfg = cfg_getter()

        # Handle batch requests
        if isinstance(body, list):
            responses = []
            for item in body:
                resp = _handle_rpc(item, cfg)
                if resp is not None:
                    responses.append(resp)
            return JSONResponse(responses if responses else {})

        resp = _handle_rpc(body, cfg)
        if resp is None:
            return JSONResponse({}, status_code=202)
        return JSONResponse(resp)

    @app.get("/mcp")
    async def mcp_sse_not_supported():
        """GET /mcp returns server info for discovery."""
        return JSONResponse({
            "name": "orze",
            "protocol": PROTOCOL_VERSION,
            "tools": len(TOOLS),
            "message": "Use POST for MCP JSON-RPC requests.",
        })

    logger.info("MCP endpoint mounted at /mcp (%d tools)", len(TOOLS))
