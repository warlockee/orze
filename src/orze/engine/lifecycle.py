"""Startup, shutdown, and PID management for Orze.

CALLING SPEC:
    startup_checks(results_dir, cfg, hostname, instance_uuid) -> HealthMonitor
    reconcile_stale_running(cfg) -> None
    print_startup_summary(cfg) -> None
    write_pid_file(results_dir) -> Path
    remove_pid_file(pid_file_path, results_dir) -> None
    write_shutdown_heartbeat(results_dir, hostname, instance_uuid, active) -> None
    graceful_shutdown(results_dir, cfg, active, active_evals, active_roles,
                      iteration, state_dict, lake, hostname, kill_all=False) -> None
    atexit_cleanup(active, active_evals, active_roles) -> None
"""

import datetime
import json
import logging
import os
import signal
import socket
import subprocess
import time
from pathlib import Path

from orze import __version__
from orze.engine.process import _kill_pg
from orze.engine.health import fs_startup_check, cleanup_stale_locks, HealthMonitor
from orze.engine.upgrade_cleanup import check_and_clean as upgrade_check_and_clean
from orze.reporting.state import save_state
from orze.reporting.notifications import notify
from orze.core.fs import _fs_unlock, atomic_write

logger = logging.getLogger("orze")


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------

def startup_checks(results_dir: Path, cfg: dict,
                   hostname: str, instance_uuid: str) -> HealthMonitor:
    """Run pre-flight checks before entering main loop.

    Returns a HealthMonitor instance for per-iteration health checks.
    """
    logger.info("=== Startup self-checks ===")
    logger.info("orze v%s | host=%s | instance=%s | pid=%d",
                __version__, hostname, instance_uuid, os.getpid())

    # 1. Verify shared filesystem is mounted and writable
    if not fs_startup_check(results_dir):
        raise SystemExit(
            f"FATAL: Shared filesystem at {results_dir} is not "
            f"writable. Check mount status.")
    logger.info("Filesystem check OK: %s", results_dir)

    # 2. Clean up stale locks from our own hostname
    cleanup_stale_locks(results_dir, hostname)

    # 3. Scrub stale state files if orze (or orze-pro) was upgraded since
    # the last boot. Must run before the FSM / roles start so they don't
    # observe pre-upgrade one-shot triggers or a pause sentinel that the
    # old binary wrote. Silent no-op on first boot.
    upgrade_check_and_clean(results_dir)

    # 4. Initialize per-iteration health monitor
    health_monitor = HealthMonitor(results_dir)

    # 5. Detect watchdog restart marker and notify
    marker = results_dir / f".orze_watchdog_restart_{hostname}.json"
    if marker.exists():
        try:
            mdata = json.loads(marker.read_text(encoding="utf-8"))
            reason = mdata.get("reason", "unknown")
            prev_pid = mdata.get("prev_pid")
            logger.info("Watchdog restart detected: %s (prev PID %s)",
                        reason, prev_pid)
            notify("watchdog_restart", {
                "host": hostname,
                "reason": reason,
                "prev_pid": prev_pid,
                "timestamp": mdata.get("iso", ""),
            }, cfg)
            marker.unlink()
        except Exception as e:
            logger.warning("Failed to process watchdog restart marker: %s", e)

    # 6. Reconcile stale "running" ideas from prior unclean shutdown
    reconcile_stale_running(cfg)

    logger.info("=== Startup checks passed ===")

    print_startup_summary(cfg)

    return health_monitor


def reconcile_stale_running(cfg: dict) -> None:
    """Reset ideas stuck in 'running' from a prior unclean shutdown.

    Only resets ideas that were claimed by THIS host (checked via
    claim.json). Ideas claimed by other hosts are left as 'running'
    since the other machine may still be training them.
    """
    import socket as _socket
    hostname = _socket.gethostname()
    results_dir = Path(cfg.get("results_dir", "orze_results"))
    lake_path = Path(cfg.get("idea_lake_db") or results_dir / "idea_lake.db")
    if not lake_path.exists():
        return
    try:
        import sqlite3
        conn = sqlite3.connect(str(lake_path), timeout=5)
        cur = conn.execute(
            "SELECT idea_id FROM ideas WHERE status = 'running'")
        all_running = [row[0] for row in cur.fetchall()]

        mine = []
        others = []
        for idea_id in all_running:
            claim_path = results_dir / idea_id / "claim.json"
            if claim_path.exists():
                try:
                    claim = json.loads(claim_path.read_text(encoding="utf-8"))
                    if claim.get("claimed_by") == hostname:
                        mine.append(idea_id)
                    else:
                        others.append(idea_id)
                except (json.JSONDecodeError, OSError):
                    mine.append(idea_id)  # can't read claim → assume ours
            else:
                mine.append(idea_id)  # no claim.json → stale, reset

        if mine:
            placeholders = ",".join("?" for _ in mine)
            conn.execute(
                f"UPDATE ideas SET status = 'queued' WHERE idea_id IN ({placeholders})",
                mine)
            conn.commit()
            logger.info("Reconciled %d stale 'running' ideas (this host) -> queued: %s",
                        len(mine), ", ".join(mine[:10]))
        if others:
            logger.info("Kept %d 'running' ideas owned by other hosts: %s",
                        len(others), ", ".join(others[:10]))
        conn.close()
    except Exception as e:
        logger.warning("Failed to reconcile stale ideas: %s", e)


def print_startup_summary(cfg: dict) -> None:
    """Print a human-readable table of what's configured."""
    W = 60
    line = "=" * W

    # Detect .env
    env_path = None
    config_path = cfg.get("_config_path")
    if config_path:
        candidate = Path(config_path).resolve().parent / ".env"
        if candidate.is_file():
            env_path = str(candidate)
    if not env_path and (Path.cwd() / ".env").is_file():
        env_path = str(Path.cwd() / ".env")

    # Evaluation
    eval_script = cfg.get("eval_script")
    eval_on = bool(eval_script and Path(eval_script).exists())

    # Research roles
    roles = cfg.get("roles") or {}
    research_names = [
        rname for rname, rcfg in roles.items()
        if isinstance(rcfg, dict) and rcfg.get("mode") in ("research", "claude")
    ]

    # Notifications
    ncfg = cfg.get("notifications", {})
    notif_on = ncfg.get("enabled", False)
    notif_channels = [
        ch.get("type", "?") for ch in ncfg.get("channels", [])
        if isinstance(ch, dict)
    ] if notif_on else []

    # Cleanup
    cleanup_cfg = cfg.get("cleanup", {})
    cleanup_on = bool(cleanup_cfg.get("script"))
    cleanup_interval = cleanup_cfg.get("interval", 100)

    # API key check for auto-research
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    has_any_llm_key = has_anthropic or has_gemini

    lines = [
        "",
        line,
        f"  orze v{__version__} -- Startup Summary",
        line,
        f"  REQUIRED:",
        f"    train_script : {cfg.get('train_script', '?')}",
        f"    ideas_file   : {cfg.get('ideas_file', '?')}",
        f"    results_dir  : {cfg.get('results_dir', '?')}",
        "",
        f"  OPTIONAL FEATURES:",
        f"    evaluation   : {'ON  (' + str(eval_script) + ')' if eval_on else 'OFF'}",
        f"    research     : {'ON  (' + ', '.join(research_names) + ')' if research_names else 'OFF'}",
        f"    notifications: {'ON  (' + ', '.join(notif_channels) + ')' if notif_on and notif_channels else 'OFF'}",
        f"    auto-cleanup : {'ON  (every ' + str(cleanup_interval) + ' ideas)' if cleanup_on else 'OFF'}",
        f"    .env file    : {'loaded (' + env_path + ')' if env_path else 'not found'}",
        line,
        "",
    ]

    from orze.extensions import has_pro
    if has_pro():
        if research_names and not has_any_llm_key:
            lines.append(
                "  \033[33m⚠ WARNING: No ANTHROPIC_API_KEY or GEMINI_API_KEY found.\033[0m"
            )
            lines.append(
                "  \033[33m  Auto-research is configured but will not work without an API key.\033[0m"
            )
            lines.append(
                "  \033[33m  Add ANTHROPIC_API_KEY or GEMINI_API_KEY to your .env file.\033[0m"
            )
            lines.append("")
            logger.warning(
                "Auto-research is configured but no ANTHROPIC_API_KEY or "
                "GEMINI_API_KEY found — research agent will not be able to "
                "generate ideas. Add a key to .env to enable auto-research."
            )
    else:
        lines.append(
            "  \033[2m💡 Upgrade to orze-pro for AI-powered idea generation,"
            " auto-fix, and code evolution → orze.ai/pro\033[0m"
        )
        lines.append("")

    for l in lines:
        print(l)


# ---------------------------------------------------------------------------
# Shutdown
# ---------------------------------------------------------------------------

def graceful_shutdown(results_dir: Path, cfg: dict,
                      active: dict, active_evals: dict, active_roles: dict,
                      iteration: int, state_dict: dict, lake,
                      hostname: str, instance_uuid: str,
                      kill_all: bool = False) -> None:
    """Terminate roles, detach or kill training/eval, save state, clean up.

    Args:
        results_dir: Path to results directory.
        cfg: Config dict.
        active: Dict of gpu -> TrainingProcess.
        active_evals: Dict of gpu -> EvalProcess.
        active_roles: Dict of role_name -> RoleProcess.
        iteration: Current iteration number.
        state_dict: Pre-built state dict for persistence.
        lake: IdeaLake instance (or None).
        hostname: This node's hostname.
        instance_uuid: This node's instance UUID.
        kill_all: If True, kill training and eval processes too (not just
                  detach them). Used by `orze --stop` to fully stop everything.
    """
    logger.info("Shutting down gracefully (kill_all=%s)...", kill_all)

    # 0. Write "shutting_down" heartbeat so other nodes know our state
    try:
        write_shutdown_heartbeat(results_dir, hostname, instance_uuid, active)
    except Exception:
        pass

    if kill_all:
        # Kill ALL child processes: training, eval, and roles
        all_procs = []
        for gpu, tp in active.items():
            logger.info("Killing training %s on GPU %s (PID %d)",
                        tp.idea_id, gpu, tp.process.pid)
            _kill_pg(tp.process, signal.SIGTERM)
            all_procs.append(("training", tp))
        for gpu, ep in active_evals.items():
            logger.info("Killing eval %s on GPU %s (PID %d)",
                        ep.idea_id, gpu, ep.process.pid)
            _kill_pg(ep.process, signal.SIGTERM)
            all_procs.append(("eval", ep))
        for role_name, rp in active_roles.items():
            logger.info("Killing role '%s' (PID %d)",
                        role_name, rp.process.pid)
            _kill_pg(rp.process, signal.SIGTERM)
            all_procs.append(("role", rp))

        # Wait up to 10s then SIGKILL
        deadline = time.time() + 10
        for label, proc in all_procs:
            remaining = max(1, deadline - time.time())
            try:
                proc.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                logger.warning("Force killing %s (PID %d)",
                               label, proc.process.pid)
                _kill_pg(proc.process, signal.SIGKILL)
                try:
                    proc.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
            proc.close_log()
            if hasattr(proc, 'lock_dir') and proc.lock_dir:
                _fs_unlock(proc.lock_dir)
    else:
        # Default: detach training/eval, kill only roles
        for gpu, tp in active.items():
            logger.info("Detaching training %s on GPU %s (PID %d) "
                        "-- will finish in background",
                        tp.idea_id, gpu, tp.process.pid)
            tp.close_log()
        for gpu, ep in active_evals.items():
            logger.info("Detaching eval %s on GPU %s (PID %d) "
                        "-- will finish in background",
                        ep.idea_id, gpu, ep.process.pid)
            ep.close_log()
        for role_name, rp in active_roles.items():
            logger.info("Terminating role '%s' (PID %d)...",
                        role_name, rp.process.pid)
            _kill_pg(rp.process, signal.SIGTERM)

        # Wait for roles to exit (up to 10s), then SIGKILL stragglers
        deadline = time.time() + 10
        for role_name, rp in active_roles.items():
            remaining = max(1, deadline - time.time())
            try:
                rp.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                logger.warning("Force killing role '%s' (PID %d)",
                               role_name, rp.process.pid)
                _kill_pg(rp.process, signal.SIGKILL)
                try:
                    rp.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.error("Failed to reap role '%s'", role_name)
            rp.close_log()
            _fs_unlock(rp.lock_dir)

    # 3. Write shutdown sentinel (tells the watchdog not to restart us)
    sentinel = results_dir / ".orze_shutdown"
    try:
        sentinel.write_text(
            f"pid={os.getpid()} iteration={iteration} "
            f"time={datetime.datetime.now().isoformat()}\n",
            encoding="utf-8",
        )
    except Exception:
        pass

    # 4. Save state for restart recovery
    save_state(results_dir, state_dict)

    # 5. Notify (best effort)
    try:
        notify("shutdown", {
            "host": hostname,
            "message": (f"Graceful shutdown after iteration "
                        f"{iteration}"),
        }, cfg)
    except Exception:
        pass

    # 6. Close IdeaLake (flushes WAL on shared filesystems)
    if lake:
        try:
            lake.close()
        except Exception:
            pass

    # 7. Clean up PID file
    remove_pid_file(results_dir / f".orze.pid.{hostname}", results_dir)

    logger.info("Shutdown complete. State saved at iteration %d. "
                "%d training and %d eval process(es) detached.",
                iteration, len(active), len(active_evals))


def atexit_cleanup(active: dict, active_evals: dict,
                   active_roles: dict) -> None:
    """Last-resort cleanup: kill all tracked child process groups."""
    for gpu, tp in list(active.items()):
        _kill_pg(tp.process, signal.SIGKILL)
        tp.close_log()
    for gpu, ep in list(active_evals.items()):
        _kill_pg(ep.process, signal.SIGKILL)
        ep.close_log()
    for role_name, rp in list(active_roles.items()):
        _kill_pg(rp.process, signal.SIGKILL)
        rp.close_log()


def write_shutdown_heartbeat(results_dir: Path, hostname: str,
                             instance_uuid: str, active: dict) -> None:
    """Write a final heartbeat marking this node as shutting_down."""
    pid = os.getpid()
    heartbeat = {
        "host": hostname,
        "pid": pid,
        "timestamp": datetime.datetime.now().isoformat(),
        "epoch": time.time(),
        "status": "shutting_down",
        "active": [
            {
                "idea_id": tp.idea_id,
                "gpu": tp.gpu,
                "elapsed_min": round((time.time() - tp.start_time) / 60, 1),
                "detached": True,
            }
            for tp in active.values()
        ],
        "free_gpus": [],
        "orze_version": __version__,
        "instance_uuid": instance_uuid,
    }
    atomic_write(results_dir / f"_host_{hostname}_{pid}.json",
                 json.dumps(heartbeat, indent=2))


# ---------------------------------------------------------------------------
# PID file management
# ---------------------------------------------------------------------------

def write_pid_file(results_dir: Path) -> Path:
    """Write host-specific PID file for clean stop via --stop or kill.

    Returns the host-specific PID file path (caller should store it for
    later removal).
    """
    hostname = socket.gethostname()
    pid_file = results_dir / f".orze.pid.{hostname}"
    pid_file.write_text(str(os.getpid()), encoding="utf-8")
    # Legacy single PID file (for backward compat)
    legacy = results_dir / ".orze.pid"
    legacy.write_text(str(os.getpid()), encoding="utf-8")
    return pid_file


def remove_pid_file(pid_file_path, results_dir: Path) -> None:
    """Remove PID files on exit."""
    for f in [pid_file_path, results_dir / ".orze.pid"]:
        try:
            if f and f.exists():
                f.unlink()
        except Exception:
            pass
