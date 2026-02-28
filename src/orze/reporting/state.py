import os
import json
import logging
import socket
import time
import datetime
import platform
import shutil
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
from orze import __version__
from orze.core.fs import atomic_write
from orze.hardware.gpu import _query_gpu_details

TrainingProcess = Any  # type alias to avoid circular import

logger = logging.getLogger("orze")


def _parse_version(v: str) -> Tuple[int, ...]:
    """Parse a semver string into a tuple of ints. Returns (0,) on failure."""
    try:
        return tuple(int(x) for x in v.split(".")[:3])
    except (ValueError, AttributeError):
        return (0,)


def write_host_heartbeat(results_dir: Path, hostname: str,
                         active, free_gpus: list):
    """Write per-host heartbeat file with active processes and free GPUs."""
    pid = os.getpid()
    now = time.time()
    heartbeat = {
        "host": hostname,
        "pid": pid,
        "timestamp": datetime.datetime.now().isoformat(),
        "epoch": now,
        "orze_version": __version__,
        "active": [
            {
                "idea_id": tp.idea_id,
                "gpu": tp.gpu,
                "elapsed_min": round((now - tp.start_time) / 60, 1),
            }
            for tp in active.values()
        ],
        "free_gpus": free_gpus,
        "gpu_info": _query_gpu_details(),
        "os": f"{platform.system()} {platform.release()}",
    }
    atomic_write(results_dir / f"_host_{hostname}_{pid}.json",
                 json.dumps(heartbeat, indent=2))


def _read_all_heartbeats(results_dir: Path,
                         stale_seconds: float = 900) -> list:
    """Read heartbeat files, keeping only the freshest per host.
    Removes superseded heartbeat files (older duplicates for the same host).
    Only removes stale files after 2x stale_seconds to avoid premature cleanup
    during long iterations on Lustre."""
    now = time.time()
    # Collect all valid heartbeats, keyed by host
    by_host: dict = {}  # host -> (epoch, hb_dict, hb_path)
    superseded_paths = []
    stale_paths = []
    for hb_path in results_dir.glob("_host_*.json"):
        try:
            hb = json.loads(hb_path.read_text(encoding="utf-8"))
            age = now - hb.get("epoch", 0)
            host = hb.get("host", "unknown")
            epoch = hb.get("epoch", 0)
            if age > stale_seconds * 2:
                # Truly dead — safe to remove
                stale_paths.append(hb_path)
                continue
            prev = by_host.get(host)
            if prev is None or epoch > prev[0]:
                if prev:
                    superseded_paths.append(prev[2])
                by_host[host] = (epoch, hb, hb_path)
            else:
                superseded_paths.append(hb_path)
        except Exception:
            try:
                if now - hb_path.stat().st_mtime > stale_seconds * 2:
                    stale_paths.append(hb_path)
            except OSError:
                pass
            continue
    # Clean up superseded (same-host duplicates) and truly stale files
    for p in superseded_paths + stale_paths:
        try:
            p.unlink()
        except OSError:
            pass
    return [v[1] for v in by_host.values()]


def check_heartbeat_versions(heartbeats: list) -> List[str]:
    """Check version compatibility across cluster heartbeats.
    Returns list of incompatible hostnames (major version mismatch).
    Logs warnings for any version mismatches."""
    my_ver = _parse_version(__version__)
    incompatible = []
    for hb in heartbeats:
        peer_ver_str = hb.get("orze_version")
        if not peer_ver_str:
            continue  # old node without version — skip silently
        peer_host = hb.get("host", "unknown")
        peer_ver = _parse_version(peer_ver_str)
        if peer_ver == my_ver:
            continue
        if peer_ver[0] != my_ver[0]:
            logger.warning("INCOMPATIBLE: %s runs orze v%s (major %d) vs our v%s (major %d). "
                           "Refusing coordination with this node.",
                           peer_host, peer_ver_str, peer_ver[0],
                           __version__, my_ver[0])
            incompatible.append(peer_host)
        else:
            logger.warning("Version mismatch: %s runs orze v%s, we run v%s",
                           peer_host, peer_ver_str, __version__)
    return incompatible


def write_status_json(results_dir: Path, iteration: int,
                      active: Dict[int, TrainingProcess],
                      free_gpus: List[int], queue_depth: int,
                      completed_count: int, failed_count: int,
                      skipped_count: int, top_results: list,
                      cfg: dict,
                      role_states: Optional[dict] = None):
    """Write machine-readable status.json for LLM agents.
    Merges heartbeats from all hosts for a combined multi-machine view."""
    disk_free_gb = 0.0
    try:
        usage = shutil.disk_usage(results_dir)
        disk_free_gb = usage.free / (1024 ** 3)
    except Exception:
        pass

    now = time.time()

    # Merge active processes from all hosts
    heartbeats = _read_all_heartbeats(results_dir)
    all_active = []
    free_gpus_by_host = {}
    for hb in heartbeats:
        host = hb.get("host", "unknown")
        for a in hb.get("active", []):
            a["host"] = host
            all_active.append(a)
        free_gpus_by_host[host] = hb.get("free_gpus", [])

    # Build per-role status
    role_states = role_states or {}
    roles_cfg = cfg.get("roles") or {}
    roles_status = {}
    for rname in roles_cfg:
        rs = role_states.get(rname, {})
        last_run = rs.get("last_run_time", 0.0)
        roles_status[rname] = {
            "enabled": True,
            "cycles": rs.get("cycles", 0),
            "last_run_min_ago": (
                round((now - last_run) / 60, 1) if last_run > 0 else None
            ),
        }

    # Backward compat: flat research_* keys
    research_rs = role_states.get("research", {})
    research_last = research_rs.get("last_run_time", 0.0)

    hostname = socket.gethostname()
    status = {
        "timestamp": datetime.datetime.now().isoformat(),
        "host": hostname,
        "iteration": iteration,
        "active": all_active,
        "free_gpus": free_gpus,
        "free_gpus_by_host": free_gpus_by_host,
        "queue_depth": queue_depth,
        "completed": completed_count,
        "failed": failed_count,
        "skipped": skipped_count,
        "disk_free_gb": round(disk_free_gb, 1),
        "top_results": top_results[:10],
        "roles": roles_status,
        "research_enabled": "research" in roles_cfg,
        "research_cycles": research_rs.get("cycles", 0),
        "last_research_min_ago": (
            round((now - research_last) / 60, 1) if research_last > 0
            else None
        ),
    }

    atomic_write(results_dir / "status.json", json.dumps(status, indent=2))


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------


def load_state(results_dir: Path) -> dict:
    """Load orchestrator state from checkpoint (per-host).
    Falls back to legacy .orze_state.json for backward compat.
    Migrates legacy flat research_* keys into roles dict."""
    hostname = socket.gethostname()
    path = results_dir / f".orze_state_{hostname}.json"

    if not path.exists():
        legacy = results_dir / ".orze_state.json"
        if legacy.exists():
            path = legacy

    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            logger.warning("Corrupt state file, starting fresh")
            return {"iteration": 0, "failure_counts": {}, "roles": {}}

        # Migrate legacy flat research state into roles dict
        if "roles" not in state and "research_cycles" in state:
            state["roles"] = {
                "research": {
                    "cycles": state.pop("research_cycles", 0),
                    "last_run_time": state.pop("last_research_time", 0.0),
                }
            }
        state.setdefault("roles", {})
        return state
    return {"iteration": 0, "failure_counts": {}, "roles": {}}


def save_state(results_dir: Path, state: dict):
    """Save orchestrator state for restart recovery (per-host)."""
    hostname = socket.gethostname()
    atomic_write(results_dir / f".orze_state_{hostname}.json",
                 json.dumps(state, indent=2))

