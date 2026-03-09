import atexit
import datetime
import json
import logging
import os
import re
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from orze import __version__
from orze.engine.process import (
    TrainingProcess, EvalProcess, RoleProcess,
    _new_process_group, _kill_pg,
    run_pre_script,
)
from orze.engine.scheduler import (
    claim, get_unclaimed, cleanup_orphans, _count_statuses, run_cleanup,
)
from orze.engine.launcher import (
    launch, check_active, _format_args, _get_checkpoint_dir, _write_failure,
)
from orze.engine.evaluator import (
    launch_eval, check_active_evals, run_eval, run_post_scripts,
)
from orze.engine.health import (
    check_disk_space, fs_startup_check, fs_check_writable,
    cleanup_stale_locks, HealthMonitor,
)
from orze.engine.roles import check_active_roles
from orze.engine.failure import (
    _record_failure, get_skipped_ideas, _try_executor_fix, _reset_idea_for_retry,
)
from orze.core.fs import _fs_lock, _fs_unlock, atomic_write, deep_get
from orze.core.ideas import parse_ideas, expand_sweeps
from orze.core.config import _validate_config
from orze.reporting.state import (
    load_state, save_state, write_host_heartbeat, _read_all_heartbeats,
    write_status_json, check_heartbeat_versions,
)
from orze.reporting.leaderboard import (
    update_report, _resolve_primary_metric, write_admin_cache,
)
from orze.reporting.notifications import notify
from orze.hardware.gpu import get_gpu_memory_used, _eval_already_running

logger = logging.getLogger("orze")


class Orze:
    def __init__(self, gpu_ids: List[int], cfg: dict, once: bool = False):
        self.gpu_ids = gpu_ids
        self.cfg = cfg
        self.once = once
        self.results_dir = Path(cfg["results_dir"])
        self.active: Dict[int, TrainingProcess] = {}
        self.active_evals: Dict[int, EvalProcess] = {}
        self.active_roles: Dict[str, RoleProcess] = {}
        self.pending_evals: list = []
        self.running = True
        self._hostname = socket.gethostname()
        self._instance_uuid = uuid.uuid4().hex[:12]
        self._incompatible_hosts: set = set()

        # Validate config on startup
        config_errors, config_warnings = _validate_config(cfg)
        for err in config_errors:
            logger.error("Config error: %s", err)
        if config_errors:
            raise SystemExit(f"Invalid config: {len(config_errors)} error(s) — see log above")
        for warn in config_warnings:
            logger.warning("Config: %s", warn)

        state = load_state(self.results_dir)
        self.iteration = state.get("iteration", 0)
        self.failure_counts = state.get("failure_counts", {})
        self.fix_counts: Dict[str, int] = state.get("fix_counts", {})

        # Per-role agent state: {role_name: {"cycles": int, "last_run_time": float}}
        self.role_states: Dict[str, dict] = state.get("roles", {})
        self._best_idea_id: Optional[str] = state.get("best_idea_id")
        self._start_time: float = time.time()
        self._last_heartbeat: float = 0.0
        self._hb_completed_count: int = 0  # for heartbeat rate calc
        self._last_milestone: int = 0      # last milestone boundary hit
        self._last_disk_warning: float = 0.0
        self._last_upgrade_check: float = 0.0
        self._pending_upgrade: Optional[str] = None

        # Initialize Idea Lake for archival
        try:
            from orze.idea_lake import IdeaLake
            lake_path = Path(cfg.get("ideas_file", "ideas.md")).parent / "idea_lake.db"

            # MOUNT INTEGRITY CHECK: Prevents split-brain leaderboard corruption.
            # Only warn about unmounted drive if results dir already has experiments
            # (indicating this is NOT a first run). On first run, lake doesn't exist yet.
            if not lake_path.exists():
                res_dir = Path(cfg.get("results_dir", "results"))
                has_prior_results = (res_dir.exists() and
                    any(d.is_dir() and d.name.startswith("idea-") for d in res_dir.iterdir())
                    if res_dir.exists() else False)
                if has_prior_results:
                    logger.error("idea_lake.db not found but results exist! "
                                 "Shared drive might not be mounted.")
                    if (cfg.get("notifications") or {}).get("enabled"):
                        logger.error("DISABLING NOTIFICATIONS: Local view may be inconsistent.")
                        cfg["notifications"]["enabled"] = False
                else:
                    logger.info("Creating idea_lake.db (first run)")

            self.lake = IdeaLake(str(lake_path))
            logger.info("Idea Lake initialized: %d archived ideas", self.lake.count())

            # Sanity check: if lake has many ideas but results dir is nearly empty,
            # this may indicate a mount failure (e.g., FSX not mounted).
            min_expected = cfg.get("min_expected_results", 0)
            if min_expected > 0:
                res_dir = Path(cfg.get("results_dir", "results"))
                if res_dir.exists():
                    n_results = len([d for d in res_dir.iterdir()
                                     if d.is_dir() and d.name.startswith("idea-")])
                    if n_results < min_expected and not once:
                        logger.warning("Results dir has only %d entries (expected >= %d). "
                                       "Possible mount failure.", n_results, min_expected)
                        if (cfg.get("notifications") or {}).get("enabled"):
                            logger.error("Disabling notifications to prevent report corruption.")
                            cfg["notifications"]["enabled"] = False
        except Exception as exc:
            logger.warning("Idea Lake not available: %s", exc)
            self.lake = None

        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Event used to interrupt poll sleep instantly on shutdown signal
        self._stop_event = threading.Event()

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        # Safety net: atexit kills all children even if signal handler
        # can't run (e.g. unhandled exception, sys.exit from 2nd signal)
        atexit.register(self._atexit_cleanup)

    def _atexit_cleanup(self):
        """Last-resort cleanup: kill all tracked child process groups."""
        for gpu, tp in list(self.active.items()):
            _kill_pg(tp.process, signal.SIGKILL)
            tp.close_log()
        for gpu, ep in list(self.active_evals.items()):
            _kill_pg(ep.process, signal.SIGKILL)
            ep.close_log()
        for role_name, rp in list(self.active_roles.items()):
            _kill_pg(rp.process, signal.SIGKILL)
            rp.close_log()

    def _startup_checks(self):
        """Run pre-flight checks before entering main loop."""
        logger.info("=== Startup self-checks ===")
        logger.info("orze v%s | host=%s | instance=%s | pid=%d",
                     __version__, self._hostname, self._instance_uuid,
                     os.getpid())

        # 1. Verify shared filesystem is mounted and writable
        if not fs_startup_check(self.results_dir):
            raise SystemExit(
                f"FATAL: Shared filesystem at {self.results_dir} is not "
                f"writable. Check mount status.")
        logger.info("Filesystem check OK: %s", self.results_dir)

        # 2. Clean up stale locks from our own hostname
        cleanup_stale_locks(self.results_dir, self._hostname)

        # 3. Initialize per-iteration health monitor
        self._health_monitor = HealthMonitor(self.results_dir)

        # 4. Detect watchdog restart marker and notify
        marker = self.results_dir / f".orze_watchdog_restart_{self._hostname}.json"
        if marker.exists():
            try:
                import json as _json_marker
                mdata = _json_marker.loads(marker.read_text(encoding="utf-8"))
                reason = mdata.get("reason", "unknown")
                prev_pid = mdata.get("prev_pid")
                logger.info("Watchdog restart detected: %s (prev PID %s)", reason, prev_pid)
                notify("watchdog_restart", {
                    "host": self._hostname,
                    "reason": reason,
                    "prev_pid": prev_pid,
                    "timestamp": mdata.get("iso", ""),
                }, self.cfg)
                marker.unlink()
            except Exception as e:
                logger.warning("Failed to process watchdog restart marker: %s", e)

        logger.info("=== Startup checks passed ===")

        self._print_startup_summary()

    def _print_startup_summary(self):
        """Print a human-readable table of what's configured."""
        cfg = self.cfg
        W = 60
        line = "=" * W

        # Detect .env
        from pathlib import Path as _P
        env_path = None
        config_path = cfg.get("_config_path")
        if config_path:
            candidate = _P(config_path).resolve().parent / ".env"
            if candidate.is_file():
                env_path = str(candidate)
        if not env_path and (_P.cwd() / ".env").is_file():
            env_path = str(_P.cwd() / ".env")

        # Evaluation
        eval_script = cfg.get("eval_script")
        eval_on = bool(eval_script and _P(eval_script).exists())

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

        lines = [
            "",
            line,
            f"  orze v{__version__} — Startup Summary",
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

        for l in lines:
            print(l)

    def _shutdown(self, signum, frame):
        """Signal handler — sets flag and wakes poll sleep immediately."""
        if not self.running:
            # Second signal = force exit
            logger.warning("Forced exit (second signal).")
            sys.exit(1)
        logger.info("Received signal %d, shutting down...", signum)
        self.running = False
        self._stop_event.set()  # wake from poll sleep instantly

    def _graceful_shutdown(self, kill_all: bool = False):
        """Terminate roles, detach or kill training/eval, save state, clean up.

        Args:
            kill_all: If True, kill training and eval processes too (not just
                      detach them). Used by `orze --stop` to fully stop everything.
        """
        logger.info("Shutting down gracefully (kill_all=%s)...", kill_all)

        # 0. Write "shutting_down" heartbeat so other nodes know our state
        try:
            self._write_shutdown_heartbeat()
        except Exception:
            pass

        if kill_all:
            # Kill ALL child processes: training, eval, and roles
            all_procs = []
            for gpu, tp in self.active.items():
                logger.info("Killing training %s on GPU %d (PID %d)",
                            tp.idea_id, gpu, tp.process.pid)
                _kill_pg(tp.process, signal.SIGTERM)
                all_procs.append(("training", tp))
            for gpu, ep in self.active_evals.items():
                logger.info("Killing eval %s on GPU %d (PID %d)",
                            ep.idea_id, gpu, ep.process.pid)
                _kill_pg(ep.process, signal.SIGTERM)
                all_procs.append(("eval", ep))
            for role_name, rp in self.active_roles.items():
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
            for gpu, tp in self.active.items():
                logger.info("Detaching training %s on GPU %d (PID %d) "
                            "— will finish in background",
                            tp.idea_id, gpu, tp.process.pid)
                tp.close_log()
            for gpu, ep in self.active_evals.items():
                logger.info("Detaching eval %s on GPU %d (PID %d) "
                            "— will finish in background",
                            ep.idea_id, gpu, ep.process.pid)
                ep.close_log()
            for role_name, rp in self.active_roles.items():
                logger.info("Terminating role '%s' (PID %d)...",
                            role_name, rp.process.pid)
                _kill_pg(rp.process, signal.SIGTERM)

            # Wait for roles to exit (up to 10s), then SIGKILL stragglers
            deadline = time.time() + 10
            for role_name, rp in self.active_roles.items():
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

        # 3. Write shutdown sentinel (tells bug_fixer not to restart us)
        sentinel = self.results_dir / ".orze_shutdown"
        try:
            sentinel.write_text(
                f"pid={os.getpid()} iteration={self.iteration} "
                f"time={datetime.datetime.now().isoformat()}\n",
                encoding="utf-8",
            )
        except Exception:
            pass

        # 4. Save state for restart recovery
        save_state(self.results_dir, {
            "iteration": self.iteration,
            "failure_counts": self.failure_counts,
            "fix_counts": self.fix_counts,
            "roles": self.role_states,
            "best_idea_id": self._best_idea_id,
        })

        # 5. Notify (best effort)
        try:
            notify("shutdown", {
                "host": self._hostname,
                "message": (f"Graceful shutdown after iteration "
                            f"{self.iteration}"),
            }, self.cfg)
        except Exception:
            pass

        # 6. Close IdeaLake (flushes WAL on shared filesystems)
        if self.lake:
            try:
                self.lake.close()
            except Exception:
                pass

        # 7. Clean up PID file
        self._remove_pid_file()

        logger.info("Shutdown complete. State saved at iteration %d. "
                     "%d training and %d eval process(es) detached.",
                     self.iteration, len(self.active), len(self.active_evals))

    def _write_shutdown_heartbeat(self):
        """Write a final heartbeat marking this node as shutting_down."""
        import json as _json
        pid = os.getpid()
        heartbeat = {
            "host": self._hostname,
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
                for tp in self.active.values()
            ],
            "free_gpus": [],
            "orze_version": __version__,
            "instance_uuid": self._instance_uuid,
        }
        atomic_write(self.results_dir / f"_host_{self._hostname}_{pid}.json",
                     _json.dumps(heartbeat, indent=2))

    def _check_cluster_versions(self):
        """Check version compatibility with other nodes. Updates _incompatible_hosts."""
        heartbeats = _read_all_heartbeats(self.results_dir)
        self._incompatible_hosts = set(check_heartbeat_versions(heartbeats))
        return heartbeats

    def _build_machine_status(self) -> list:
        """Build machine status from heartbeats for report notifications."""
        heartbeats = _read_all_heartbeats(self.results_dir)
        # Check versions on every heartbeat read
        self._incompatible_hosts = set(check_heartbeat_versions(heartbeats))
        machines = []
        for hb in heartbeats:
            host = hb.get("host", "unknown")
            active_list = hb.get("active", [])
            free_list = hb.get("free_gpus", [])
            gpus_busy = len(active_list)
            gpus_total = gpus_busy + len(free_list)
            util = round(gpus_busy / gpus_total * 100) if gpus_total else 0
            machines.append({
                "host": host,
                "gpus_busy": gpus_busy,
                "gpus_total": gpus_total,
                "utilization": util,
            })
        return machines

    def _process_notifications(self, finished: list,
                               completed_rows: list, ideas: dict,
                               counts: dict):
        """Fire notifications for finished experiments and new bests. Never raises."""
        try:
            cfg = self.cfg
            ncfg = cfg.get("notifications") or {}
            if not ncfg.get("enabled", False):
                logger.debug("Notifications disabled")
                return

            if not finished:
                return

            logger.info("Processing notifications for %d finished items",
                        len(finished))

            primary = cfg["report"].get("primary_metric", "test_accuracy")

            # Build rank lookup and leaderboard from sorted completed_rows
            rank_lookup = {}
            leaderboard = []
            for rank, r in enumerate(completed_rows, 1):
                rank_lookup[r["id"]] = rank
                if rank <= 10:
                    leaderboard.append({
                        "id": r["id"],
                        "title": r.get("title", r["id"]),
                        "value": r.get("primary_val"),
                    })

            # Build view leaderboards from cached JSON files
            view_leaderboards = {}
            views = cfg.get("report", {}).get("views") or []
            for view in views:
                vname = view.get("name")
                if not vname:
                    continue
                vpath = self.results_dir / f"_leaderboard_{vname}.json"
                if vpath.exists():
                    try:
                        vdata = json.loads(vpath.read_text(encoding="utf-8"))
                        vtop = []
                        for entry in (vdata.get("top") or [])[:10]:
                            vtop.append({
                                "id": entry.get("idea_id", "?"),
                                "title": entry.get("title", ""),
                                "value": entry.get("metric_value"),
                            })
                        if vtop:
                            view_leaderboards[vname] = {
                                "title": vdata.get("title", vname),
                                "entries": vtop,
                            }
                    except (json.JSONDecodeError, OSError):
                        pass

            # Build a lookup from completed_rows for metric values
            row_lookup = {r["id"]: r for r in completed_rows}

            # Notify for each finished experiment
            for idea_id, gpu in finished:
                m_path = self.results_dir / idea_id / "metrics.json"
                if not m_path.exists():
                    continue
                try:
                    m = json.loads(m_path.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                    continue

                status = m.get("status", "UNKNOWN")
                title = ideas.get(idea_id, {}).get("title", idea_id)

                if status == "COMPLETED":
                    # Use primary_val from report rows, fall back to
                    # reading eval report directly (for just-completed evals)
                    row = row_lookup.get(idea_id, {})
                    metric_val = row.get("primary_val") or m.get(primary)
                    if metric_val is None:
                        # Report rows are stale — read eval output directly
                        eval_file = cfg.get("eval_output", "eval_report.json")
                        eval_path = self.results_dir / idea_id / eval_file
                        if eval_path.exists():
                            try:
                                ed = json.loads(eval_path.read_text(
                                    encoding="utf-8"))
                                # Derive metric path from report columns config
                                metric_val = _resolve_primary_metric(
                                    cfg, eval_file, ed)
                            except (json.JSONDecodeError, OSError,
                                    KeyError, UnicodeDecodeError):
                                pass

                    # SYSTEMATIC FIX: No Last Resort fallback.
                    # We only report what we have verified.

                    if metric_val is None:
                        logger.warning(
                            "Notification for %s has metric_val=None "
                            "(row_pv=%s, m.get(%s)=%s, eval_exists=%s)",
                            idea_id,
                            row.get("primary_val"),
                            primary, m.get(primary),
                            (self.results_dir / idea_id /
                             cfg.get("eval_output", "eval_report.json")
                             ).exists())
                    t_time = m.get("training_time") or None
                    # Format metric to 4 decimal places
                    fmt_val = (f"{metric_val:.4f}"
                               if isinstance(metric_val, (int, float))
                               else metric_val)
                    notify("completed", {
                        "idea_id": idea_id, "title": title,
                        "metric_name": primary,
                        "metric_value": fmt_val,
                        "training_time": t_time,
                        "rank": rank_lookup.get(idea_id, "?"),
                        "leaderboard": leaderboard,
                        "view_leaderboards": view_leaderboards,
                    }, cfg)
                elif status == "FAILED":
                    notify("failed", {
                        "idea_id": idea_id, "title": title,
                        "error": m.get("error", "unknown"),
                        "leaderboard": leaderboard,
                        "view_leaderboards": view_leaderboards,
                    }, cfg)

                # Auto-archive to Idea Lake
                if self.lake and status in ("COMPLETED", "FAILED"):
                    try:
                        idea_data = ideas.get(idea_id, {})
                        config_yaml = ""
                        raw_md = idea_data.get("raw", "")
                        if idea_data.get("config"):
                            config_yaml = yaml.dump(idea_data["config"],
                                                    default_flow_style=False)
                        # Load eval metrics if available
                        eval_metrics = {}
                        eval_file = cfg.get("eval_output", "eval_report.json")
                        eval_path = self.results_dir / idea_id / eval_file
                        if eval_path.exists():
                            try:
                                ed = json.loads(eval_path.read_text(encoding="utf-8"))
                                em = ed.get("metrics", {})
                                # Store all report column metrics dynamically
                                report_cols = cfg.get("report", {}).get(
                                    "columns", [])
                                for col in report_cols:
                                    src = col.get("source", "")
                                    key = col.get("key", "")
                                    if ":" in src:
                                        src_file, json_path = src.split(":", 1)
                                        if src_file == eval_file:
                                            val = deep_get(ed, json_path)
                                            if val is not None:
                                                eval_metrics[key] = val
                                    elif key and key in em:
                                        eval_metrics[key] = em[key]
                            except (json.JSONDecodeError, OSError):
                                pass
                        # Ensure status is updated in DB if archived
                        def _raw_field(field):
                            m = re.search(rf"\*\*{re.escape(field)}\*\*:\s*(.+)", raw_md)
                            return m.group(1).strip() if m else None
                        self.lake.insert(
                            idea_id, title, config_yaml, raw_md,
                            eval_metrics=eval_metrics or None,
                            status=status.lower(),
                            priority=idea_data.get("priority", "medium"),
                            category=_raw_field("Category"),
                            parent=_raw_field("Parent"),
                            hypothesis=_raw_field("Hypothesis"),
                        )
                    except Exception as exc:
                        logger.warning("Failed to archive %s to lake: %s",
                                       idea_id, exc)

            # New best detection
            if completed_rows:
                current_best = completed_rows[0]["id"]
                if (self._best_idea_id is not None
                        and current_best != self._best_idea_id):
                    best_val = completed_rows[0].get("primary_val")
                    fmt_best = (f"{best_val:.4f}"
                                if isinstance(best_val, (int, float))
                                else best_val)
                    notify("new_best", {
                        "idea_id": current_best,
                        "title": completed_rows[0]["title"],
                        "metric_name": primary,
                        "metric_value": fmt_best,
                        "prev_best_id": self._best_idea_id,
                        "leaderboard": leaderboard,
                        "view_leaderboards": view_leaderboards,
                    }, cfg)
                self._best_idea_id = current_best

            # Periodic report summary
            report_interval = ncfg.get("report_interval", 0)
            if report_interval > 0:
                last_report = getattr(self, "_last_report_notify", 0.0)
                if time.time() - last_report >= report_interval:
                    notify("report", {
                        "title": cfg["report"].get("title", "Report"),
                        "completed": counts.get("COMPLETED", 0),
                        "failed": counts.get("FAILED", 0),
                        "active_count": len(self.active),
                        "queued": counts.get("QUEUED", 0),
                        "metric_name": primary,
                        "leaderboard": leaderboard,
                        "view_leaderboards": view_leaderboards,
                        "machines": self._build_machine_status(),
                    }, cfg)
                    self._last_report_notify = time.time()

        except Exception as e:
            logger.warning("Notification processing error: %s", e)

    def _run_role_step(self, role_name: str, role_cfg: dict):
        """Launch agent role if not running and cooldown elapsed (non-blocking).

        Supports two modes:
          - mode: script  — run a Python script
          - mode: claude  — run Claude CLI with a rules/prompt file
        """
        # Skip if already running
        if role_name in self.active_roles:
            return

        mode = role_cfg.get("mode", "script")
        if mode == "script" and not role_cfg.get("script"):
            return
        if mode == "claude" and not role_cfg.get("rules_file"):
            return
        if mode == "research" and not role_cfg.get("backend"):
            return

        # Per-role cooldown (with adaptive producer-consumer matching)
        role_state = self.role_states.setdefault(
            role_name, {"cycles": 0, "last_run_time": 0.0})
        cooldown = role_cfg.get("cooldown", 300)
        elapsed = time.time() - role_state["last_run_time"]

        # Adaptive cooldown: if queue is nearly empty, skip cooldown to
        # keep GPUs fed. Only applies to the research role.
        queue_starving = False
        if role_name == "research" and elapsed >= 60:
            try:
                ideas = parse_ideas(self.cfg["ideas_file"])
                skipped = get_skipped_ideas(
                    self.failure_counts,
                    self.cfg.get("max_idea_failures", 0))
                n_unclaimed = len(get_unclaimed(
                    ideas, self.results_dir, skipped))
                n_gpus = len(self.gpu_ids)

                # --- Hard queue cap: skip research entirely ---
                max_queue = self.cfg.get("max_queue_size", 500)
                if n_unclaimed > max_queue:
                    logger.info(
                        "Queue full (%d > %d) — skipping research",
                        n_unclaimed, max_queue)
                    return

                # --- Adaptive cooldown: scale with queue depth ---
                # When queue is deep, slow down to save API costs.
                # When shallow, keep configured cooldown or trigger early.
                if n_unclaimed < n_gpus * 2:
                    queue_starving = True
                    logger.info(
                        "Queue low (%d unclaimed, %d GPUs) — "
                        "triggering research early", n_unclaimed, n_gpus)
                elif n_unclaimed > n_gpus * 8:
                    # Queue is deep — double the cooldown
                    cooldown = cooldown * 2
                    logger.debug(
                        "Queue deep (%d unclaimed, %d GPUs) — "
                        "cooldown extended to %ds", n_unclaimed, n_gpus,
                        cooldown)

                # --- Convergence slowdown ---
                # If primary metric hasn't improved in N completed ideas,
                # multiply cooldown. Uses completed count as a stable
                # monotonic signal (not wall-clock time).
                patience = self.cfg.get("convergence_patience", 0)
                if patience > 0:
                    best_val = role_state.get("_best_metric_val")
                    best_at = role_state.get("_best_metric_at", 0)
                    counts = _count_statuses(ideas, self.results_dir)
                    n_completed = counts.get("COMPLETED", 0)

                    # Read current best from completed rows
                    primary = self.cfg["report"].get(
                        "primary_metric", "test_accuracy")
                    sort_desc = self.cfg["report"].get(
                        "sort", "descending") == "descending"
                    cur_best = None
                    for d in self.results_dir.iterdir():
                        if not d.is_dir() or not d.name.startswith("idea-"):
                            continue
                        mp = d / "metrics.json"
                        if not mp.exists():
                            continue
                        try:
                            m = json.loads(
                                mp.read_text(encoding="utf-8"))
                            if m.get("status") != "COMPLETED":
                                continue
                            v = m.get(primary)
                            if v is None:
                                continue
                            if cur_best is None:
                                cur_best = v
                            elif sort_desc and v > cur_best:
                                cur_best = v
                            elif not sort_desc and v < cur_best:
                                cur_best = v
                        except Exception:
                            continue

                    if cur_best is not None:
                        improved = False
                        if best_val is None:
                            improved = True
                        elif sort_desc and cur_best > best_val:
                            improved = True
                        elif not sort_desc and cur_best < best_val:
                            improved = True

                        if improved:
                            role_state["_best_metric_val"] = cur_best
                            role_state["_best_metric_at"] = n_completed
                        elif n_completed - best_at >= patience:
                            stale = n_completed - best_at
                            multiplier = 1 + (stale // patience)
                            cooldown = int(cooldown * multiplier)
                            logger.info(
                                "Convergence: no improvement in %d ideas "
                                "(best=%s at %d) — cooldown %ds",
                                stale, best_val, best_at, cooldown)
            except Exception:
                pass

        if elapsed < cooldown and not queue_starving:
            return

        timeout = role_cfg.get("timeout", 600)

        # Per-role cross-machine lock
        lock_dir = self.results_dir / f"_{role_name}_lock"
        if not _fs_lock(lock_dir, stale_seconds=timeout + 60):
            logger.debug("%s lock held by another host, skipping", role_name)
            return

        # Template variables (shared across all roles)
        ideas = parse_ideas(self.cfg["ideas_file"])
        counts = _count_statuses(ideas, self.results_dir)
        template_vars = {
            "ideas_file": self.cfg["ideas_file"],
            "results_dir": str(self.results_dir),
            "cycle": role_state["cycles"] + 1,
            "gpu_count": len(self.gpu_ids),
            "completed": counts.get("COMPLETED", 0),
            "queued": counts.get("QUEUED", 0),
            "role_name": role_name,
        }

        # Build command based on mode
        if mode == "claude":
            cmd = self._build_claude_cmd(role_cfg, template_vars)
            if not cmd:
                _fs_unlock(lock_dir)
                return
        elif mode == "research":
            cmd = self._build_research_cmd(role_cfg, template_vars)
        else:
            python = self.cfg.get("python", sys.executable)
            cmd = [python, role_cfg["script"]]
            cmd.extend(_format_args(role_cfg.get("args") or [], template_vars))

        # Environment
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)  # Allow nested Claude CLI sessions
        env.pop("CLAUDE_CODE_ENTRYPOINT", None)
        for k, v in (self.cfg.get("train_extra_env") or {}).items():
            env[k] = str(v)
        for k, v in (role_cfg.get("env") or {}).items():
            env[k] = str(v)

        # Per-role log directory
        log_dir_name = role_cfg.get("log_dir") or f"_{role_name}_logs"
        log_dir = self.results_dir / log_dir_name
        log_dir.mkdir(parents=True, exist_ok=True)
        cycle_num = role_state["cycles"] + 1
        log_path = log_dir / f"cycle_{cycle_num:03d}.log"

        logger.info("Running %s [%s] (cycle %d)...",
                     role_name, mode, cycle_num)

        # Protect ideas.md: snapshot size before research role runs
        ideas_file = Path(self.cfg.get("ideas_file", "ideas.md"))
        ideas_pre_size = 0
        ideas_pre_count = 0
        if ideas_file.exists():
            ideas_pre_size = ideas_file.stat().st_size
            ideas_pre_count = len(re.findall(
                r"^## idea-[a-z0-9]+:", ideas_file.read_text(encoding="utf-8"),
                re.MULTILINE))
            # Rotate backups: keep last 3
            # Wrapped in try/except: on shared FSX, concurrent roles can race
            # on the same backup files (each role has its own lock, but all
            # roles share ideas.md.safe*). A TOCTOU between exists() and
            # rename() raises FileNotFoundError which must not abort the role.
            try:
                backup_base = ideas_file.with_suffix(".md.safe")
                for i in range(2, 0, -1):
                    src = Path(f"{backup_base}.{i}")
                    dst = Path(f"{backup_base}.{i + 1}")
                    if src.exists():
                        src.rename(dst)
                if backup_base.exists():
                    backup_base.rename(Path(f"{backup_base}.1"))
                import shutil
                shutil.copy2(str(ideas_file), str(backup_base))
                logger.debug("ideas.md backup: %d bytes, %d ideas",
                             ideas_pre_size, ideas_pre_count)
            except OSError as _backup_err:
                logger.debug("ideas.md backup skipped (concurrent rename race): %s",
                             _backup_err)

        # Launch non-blocking
        log_fh = None
        try:
            log_fh = open(log_path, "w", encoding="utf-8")
            proc = subprocess.Popen(
                cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
                preexec_fn=_new_process_group,
            )
            self.active_roles[role_name] = RoleProcess(
                role_name=role_name,
                process=proc,
                start_time=time.time(),
                log_path=log_path,
                timeout=timeout,
                lock_dir=lock_dir,
                cycle_num=cycle_num,
                _log_fh=log_fh,
                ideas_pre_size=ideas_pre_size,
                ideas_pre_count=ideas_pre_count,
            )
        except Exception as e:
            logger.warning("%s launch error: %s", role_name, e)
            if log_fh and not log_fh.closed:
                log_fh.close()
            _fs_unlock(lock_dir)

    def _run_role_once(self, role_name: str):
        """Run a single agent role synchronously, then exit."""
        roles = self.cfg.get("roles") or {}
        if role_name not in roles:
            logger.error("Role '%s' not found in config. Available: %s",
                         role_name, list(roles.keys()))
            return
        role_cfg = roles[role_name]
        if not isinstance(role_cfg, dict):
            logger.error("Role '%s' config is not a dict", role_name)
            return

        logger.info("Running role '%s' once...", role_name)
        self._run_role_step(role_name, role_cfg)

        # Wait for it to finish
        while role_name in self.active_roles:
            time.sleep(2)
            finished = check_active_roles(
                self.active_roles,
                ideas_file=self.cfg.get("ideas_file", "ideas.md"))
            for rn, success in finished:
                if success:
                    logger.info("Role '%s' completed successfully", rn)
                else:
                    logger.warning("Role '%s' failed", rn)

    def _run_all_roles(self):
        """Check active roles and launch new ones (non-blocking)."""
        # Check active roles
        finished = check_active_roles(
            self.active_roles,
            ideas_file=self.cfg.get("ideas_file", "ideas.md"))

        # Collect per-role results for consolidated notification
        role_contributions = {}  # role_name -> new_ideas count
        any_ideas_modified = False

        for role_name, success in finished:
            role_state = self.role_states.setdefault(
                role_name, {"cycles": 0, "last_run_time": 0.0})
            role_state["last_run_time"] = time.time()
            role_state["cycles"] = role_state.get("cycles", 0) + 1

            if success:
                role_state["consecutive_failures"] = 0
                # Output validation: warn if ideas file wasn't modified
                ideas_file = Path(self.cfg.get("ideas_file", "ideas.md"))
                ideas_modified = False
                if ideas_file.exists():
                    ideas_age = time.time() - ideas_file.stat().st_mtime
                    role_timeout = (self.cfg.get("roles") or {}).get(
                        role_name, {}).get("timeout", 600)
                    ideas_modified = ideas_age <= role_timeout
                    if not ideas_modified:
                        logger.warning("%s completed successfully but ideas file "
                                       "was not modified (last change %.0fs ago)",
                                       role_name, ideas_age)
                if ideas_modified:
                    ideas_now = parse_ideas(self.cfg["ideas_file"])
                    prev_count = role_state.get("_prev_idea_count", 0)
                    new_ideas = max(0, len(ideas_now) - prev_count)
                    role_state["_prev_idea_count"] = len(ideas_now)
                    any_ideas_modified = True
                    # Derive a short label from the role's model config
                    role_cfg = (self.cfg.get("roles") or {}).get(role_name, {})
                    model = role_cfg.get("model", role_name)
                    # Shorten model name: "claude-opus-4-6" -> "opus", "gemini-2.5-pro" -> "gemini"
                    label = model
                    for prefix in ("claude-", "gemini-"):
                        if label.startswith(prefix):
                            label = label[len(prefix):]
                            break
                    label = label.split("-")[0]  # "opus", "2.5" -> first part
                    role_contributions[label] = (
                        role_contributions.get(label, 0) + new_ideas)
                else:
                    role_contributions.setdefault(role_name, 0)
            else:
                consec = role_state.get("consecutive_failures", 0) + 1
                role_state["consecutive_failures"] = consec
                if consec >= 3:
                    logger.error("%s has failed %d consecutive times — "
                                 "check config or script", role_name, consec)

        # Send one consolidated role_summary notification
        if finished and any_ideas_modified:
            ideas_now = parse_ideas(self.cfg["ideas_file"])
            n_queued = len(get_unclaimed(ideas_now, self.results_dir, set()))
            total_new = sum(role_contributions.values())
            # Build breakdown string: "5 <opus> + 10 <gemini>"
            parts = [f"{n} <{lbl}>" for lbl, n in role_contributions.items() if n > 0]
            breakdown = " + ".join(parts) if parts else f"{total_new}"
            notify("role_summary", {
                "role": "researcher",
                "new_ideas": total_new,
                "breakdown": breakdown,
                "queued": n_queued,
            }, self.cfg)

        # Launch new roles if not running
        for role_name, role_cfg in (self.cfg.get("roles") or {}).items():
            if isinstance(role_cfg, dict):
                self._run_role_step(role_name, role_cfg)

    def _build_claude_cmd(self, research_cfg: dict,
                          template_vars: dict) -> Optional[List[str]]:
        """Build a Claude CLI command for mode: claude."""
        rules_file = research_cfg["rules_file"]
        rules_path = Path(rules_file)
        if not rules_path.exists():
            logger.warning("Research rules file not found: %s", rules_file)
            return None

        rules_content = rules_path.read_text(encoding="utf-8")

        # Substitute template vars using explicit replace (safe with literal {})
        prompt = rules_content
        for k, v in template_vars.items():
            prompt = prompt.replace(f"{{{k}}}", str(v))

        claude_bin = research_cfg.get("claude_bin") or "claude"
        cmd = [claude_bin, "-p", prompt]

        # --model (e.g., sonnet, opus, haiku)
        model = research_cfg.get("model")
        if model:
            cmd.extend(["--model", model])

        # --allowedTools (default: local tools only — add WebSearch,WebFetch in config if needed)
        allowed_tools = research_cfg.get("allowed_tools") or "Read,Write,Edit,Glob,Grep,Bash"
        cmd.extend(["--allowedTools", str(allowed_tools)])

        # --output-format
        output_format = research_cfg.get("output_format") or "text"
        cmd.extend(["--output-format", str(output_format)])

        # Any extra CLI args
        cmd.extend(_format_args(research_cfg.get("claude_args") or [],
                                template_vars))

        return cmd

    def _build_research_cmd(self, role_cfg: dict,
                            template_vars: dict) -> List[str]:
        """Build command for mode: research (built-in LLM research agent).

        Minimal config:
            research_gemini:
              mode: research
              backend: gemini       # gemini, openai, anthropic, ollama, custom
              model: gemini-2.5-flash  # optional
              endpoint: http://...  # optional, for ollama/custom
              rules_file: RULES.md  # optional, project-specific guidance
              env:
                GEMINI_API_KEY: "..."
        """
        python = self.cfg.get("python", sys.executable)
        # research_agent.py lives in the agents directory
        agent_script = Path(__file__).parent.parent / "agents" / "research.py"

        cmd = [python, str(agent_script)]
        cmd.extend(["-c", str(self.cfg.get("_config_path", "orze.yaml"))])
        cmd.extend(["--backend", role_cfg["backend"]])
        cmd.extend(["--cycle", str(template_vars["cycle"])])
        cmd.extend(["--ideas-md", str(template_vars["ideas_file"])])
        cmd.extend(["--results-dir", str(template_vars["results_dir"])])

        if role_cfg.get("model"):
            cmd.extend(["--model", str(role_cfg["model"])])
        if role_cfg.get("endpoint"):
            cmd.extend(["--endpoint", str(role_cfg["endpoint"])])
        if role_cfg.get("num_ideas"):
            cmd.extend(["--num-ideas", str(role_cfg["num_ideas"])])
        if role_cfg.get("rules_file"):
            cmd.extend(["--rules-file", str(role_cfg["rules_file"])])

        # Pass lake DB path so research agent can query historical patterns
        lake_path = Path(template_vars["ideas_file"]).parent / "idea_lake.db"
        if lake_path.exists():
            cmd.extend(["--lake-db", str(lake_path)])

        return cmd

    def _kill_orphans(self):
        """Kill orphaned train/eval processes from a previous Orze instance.

        On startup, scan for processes that match our command patterns
        (the configured train_script and eval_script) with our results dir in their
        cmdline, but whose parent is init (PPID=1) — i.e. orphans.
        Kill their entire process groups.
        """
        my_pid = os.getpid()
        results_str = str(self.results_dir)
        cfg = self.cfg
        patterns = [Path(cfg.get("train_script", "train.py")).name]
        if cfg.get("eval_script"):
            patterns.append(Path(cfg["eval_script"]).name)
        killed = 0

        try:
            for entry in os.listdir("/proc"):
                if not entry.isdigit():
                    continue
                pid = int(entry)
                if pid == my_pid:
                    continue
                try:
                    stat = Path(f"/proc/{pid}/stat").read_text()
                    ppid = int(stat.split(")")[1].split()[1])
                    if ppid != 1:
                        continue  # Not an orphan
                    cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
                    cmdline_str = cmdline.decode("utf-8", errors="replace")
                    # Check it matches our patterns AND our results dir
                    if (results_str in cmdline_str and
                            any(p in cmdline_str for p in patterns)):
                        # Kill the entire process group
                        try:
                            pgid = os.getpgid(pid)
                            os.killpg(pgid, signal.SIGKILL)
                            killed += 1
                            logger.info("Killed orphan process group %d "
                                        "(leader PID %d)", pgid, pid)
                        except (ProcessLookupError, PermissionError, OSError):
                            pass
                except (FileNotFoundError, ValueError, IndexError,
                        PermissionError, OSError):
                    continue
        except OSError:
            pass

        if killed:
            logger.info("Cleaned up %d orphaned process group(s)", killed)

    def _write_pid_file(self):
        """Write host-specific PID file for clean stop via --stop or kill."""
        hostname = socket.gethostname()
        self._pid_file = self.results_dir / f".orze.pid.{hostname}"
        self._pid_file.write_text(str(os.getpid()), encoding="utf-8")
        # Legacy single PID file (for backward compat)
        legacy = self.results_dir / ".orze.pid"
        legacy.write_text(str(os.getpid()), encoding="utf-8")

    def _remove_pid_file(self):
        """Remove PID files on exit."""
        for f in [getattr(self, "_pid_file", None),
                  self.results_dir / ".orze.pid"]:
            try:
                if f and f.exists():
                    f.unlink()
            except Exception:
                pass

    def _check_stop_all(self):
        """Check for filesystem-based stop signal (.orze_stop_all).

        This allows stopping all orze instances across machines
        that share the same results directory (e.g. on NFS/FSx).
        If the sentinel contains "kill", training/eval are killed too.
        """
        stop_file = self.results_dir / ".orze_stop_all"
        if stop_file.exists():
            try:
                content = stop_file.read_text(encoding="utf-8").strip()
            except OSError:
                content = ""
            self._stop_kill_all = "kill" in content.lower()
            logger.info("Found .orze_stop_all — shutting down (kill_all=%s)",
                        self._stop_kill_all)
            self.running = False
            return True
        return False

    def _check_disabled(self):
        """Check for persistent disable flag (.orze_disabled).

        Unlike .orze_stop_all (cleared on startup), this file persists
        and prevents Orze from starting at all. Must be manually removed
        to re-enable:  rm results/.orze_disabled
        """
        disabled_file = self.results_dir / ".orze_disabled"
        if disabled_file.exists():
            msg = disabled_file.read_text(encoding="utf-8").strip()
            logger.error("Orze is DISABLED: %s", msg)
            logger.error("Remove %s to re-enable", disabled_file)
            return True
        return False

    def _check_auto_upgrade(self):
        """Check PyPI for a newer orze version. Rate-limited. Never raises."""
        au_cfg = self.cfg.get("auto_upgrade")
        if not au_cfg:
            return
        if isinstance(au_cfg, bool):
            interval = 3600
        else:
            interval = int(au_cfg.get("interval", 3600))

        if time.time() - self._last_upgrade_check < interval:
            return
        self._last_upgrade_check = time.time()

        try:
            import urllib.request
            req = urllib.request.Request(
                "https://pypi.org/pypi/orze/json",
                headers={"User-Agent": f"orze/{__version__}"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            latest = data["info"]["version"]
        except Exception as exc:
            logger.warning("Auto-upgrade: PyPI check failed: %s", exc)
            return

        def _ver(s):
            try:
                return tuple(int(x) for x in s.split(".")[:3])
            except (ValueError, AttributeError):
                return (0,)

        if _ver(latest) > _ver(__version__):
            if self._pending_upgrade != latest:
                logger.info("Auto-upgrade: v%s available (current v%s)",
                            latest, __version__)
            self._pending_upgrade = latest
        else:
            self._pending_upgrade = None

    def _check_upgrade_sentinel(self):
        """Check if another node wrote .orze_upgrade sentinel.
        If the target version is newer, restart via os.execv (pip already done)."""
        sentinel = self.results_dir / ".orze_upgrade"
        if not sentinel.exists():
            return
        try:
            target = sentinel.read_text(encoding="utf-8").strip()
        except Exception:
            return

        def _ver(s):
            try:
                return tuple(int(x) for x in s.split(".")[:3])
            except (ValueError, AttributeError):
                return (0,)

        if _ver(target) <= _ver(__version__):
            # Already at or past this version, clean up sentinel
            try:
                sentinel.unlink(missing_ok=True)
            except OSError:
                pass
            return

        logger.info("Auto-upgrade: sentinel found — another node upgraded to v%s, restarting...", target)
        self._pending_upgrade = target
        self._do_auto_upgrade()

    def _do_auto_upgrade(self):
        """Install pending upgrade, kill everything, and restart via os.execv."""
        target = self._pending_upgrade
        logger.info("Auto-upgrade: installing orze==%s (current v%s)...",
                     target, __version__)

        # Acquire upgrade lock to prevent concurrent upgrades across processes
        upgrade_lock = self.results_dir / f"_upgrade_lock"
        if not _fs_lock(upgrade_lock, stale_seconds=300):
            logger.info("Auto-upgrade: another process is already upgrading, skipping")
            self._pending_upgrade = None
            return

        # Write shutdown sentinel so watchdog won't respawn during restart
        try:
            (self.results_dir / ".orze_shutdown").write_text(
                f"auto-upgrade to v{target}", encoding="utf-8")
        except OSError:
            pass

        try:
            notify("upgrading", {
                "host": socket.gethostname(),
                "from_version": __version__,
                "to_version": target,
                "message": f"Upgrading v{__version__} -> v{target}, restarting",
            }, self.cfg)
        except Exception:
            pass

        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", f"orze=={target}",
             "--quiet"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.error("Auto-upgrade pip install failed (rc=%d): %s",
                         result.returncode, result.stderr[:500])
            self._pending_upgrade = None
            _fs_unlock(upgrade_lock)
            # Remove shutdown sentinel on failure so watchdog can restart
            try:
                (self.results_dir / ".orze_shutdown").unlink(missing_ok=True)
            except OSError:
                pass
            return

        # Signal other nodes sharing this results_dir to restart
        try:
            (self.results_dir / ".orze_upgrade").write_text(
                target, encoding="utf-8")
        except Exception:
            pass

        _fs_unlock(upgrade_lock)

        logger.info("Auto-upgrade: killing active processes and restarting...")

        # Kill all children (training will restart on new version)
        for gpu, tp in self.active.items():
            _kill_pg(tp.process, signal.SIGTERM)
        for role_name, rp in self.active_roles.items():
            _kill_pg(rp.process, signal.SIGTERM)
        deadline = time.time() + 10
        for gpu, tp in list(self.active.items()):
            remaining = max(1, deadline - time.time())
            try:
                tp.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                _kill_pg(tp.process, signal.SIGKILL)
            tp.close_log()
        for gpu, ep in list(self.active_evals.items()):
            ep.close_log()
        for role_name, rp in list(self.active_roles.items()):
            remaining = max(1, deadline - time.time())
            try:
                rp.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                _kill_pg(rp.process, signal.SIGKILL)
            rp.close_log()
            _fs_unlock(rp.lock_dir)

        try:
            save_state(self.results_dir, {
                "iteration": self.iteration,
                "failure_counts": self.failure_counts,
                "fix_counts": self.fix_counts,
                "roles": self.role_states,
                "best_idea_id": self._best_idea_id,
            })
        except Exception as e:
            logger.warning("Auto-upgrade: state save failed: %s", e)

        if self.lake:
            try:
                self.lake.close()
            except Exception:
                pass

        self._remove_pid_file()

        config_path = str(self.cfg.get("_config_path", "orze.yaml"))
        logger.info("Auto-upgrade: restarting with v%s (config: %s)",
                     target, config_path)
        try:
            os.execv(sys.executable,
                     [sys.executable, "-m", "orze.cli", "-c", config_path])
        except OSError as exc:
            logger.error("Auto-upgrade: os.execv failed: %s — restart manually",
                         exc)
            sys.exit(1)

    def run(self):
        cfg = self.cfg
        self._write_pid_file()
        self._startup_checks()
        self._kill_orphans()
        # Check persistent disable flag (never auto-deleted)
        if self._check_disabled():
            logger.error("Exiting — Orze is disabled")
            return
        # Clear any stale shutdown sentinels from a previous run
        for sentinel_name in [".orze_shutdown", ".orze_stop_all"]:
            sentinel = self.results_dir / sentinel_name
            if sentinel.exists():
                sentinel.unlink(missing_ok=True)
        # Clear upgrade sentinel if we're already at the target version
        upgrade_sentinel = self.results_dir / ".orze_upgrade"
        if upgrade_sentinel.exists():
            try:
                target = upgrade_sentinel.read_text(encoding="utf-8").strip()
                def _ver(s):
                    try:
                        return tuple(int(x) for x in s.split(".")[:3])
                    except (ValueError, AttributeError):
                        return (0,)
                if _ver(__version__) >= _ver(target):
                    upgrade_sentinel.unlink(missing_ok=True)
            except Exception:
                pass
        logger.info("Starting orze v%s on GPUs %s (PID %d)",
                     __version__, self.gpu_ids, os.getpid())
        logger.info("Ideas: %s | Results: %s | Timeout: %ds | Poll: %ds",
                     cfg["ideas_file"], cfg["results_dir"],
                     cfg["timeout"], cfg["poll"])
        for rname, rcfg in (cfg.get("roles") or {}).items():
            if not isinstance(rcfg, dict):
                continue
            rmode = rcfg.get("mode", "script")
            rtarget = (rcfg.get("rules_file") if rmode == "claude"
                       else rcfg.get("script"))
            if rtarget:
                logger.info("Role '%s' [%s]: %s (cooldown: %ds, timeout: %ds)",
                            rname, rmode, rtarget,
                            rcfg.get("cooldown", 300),
                            rcfg.get("timeout", 600))

        # Lifecycle notification: started
        n_roles = len([r for r in (cfg.get("roles") or {}).values()
                       if isinstance(r, dict)])
        notify("started", {
            "host": socket.gethostname(),
            "message": (f"v{__version__} | {len(self.gpu_ids)} GPUs | "
                        f"{n_roles} roles | pid {os.getpid()}"),
        }, cfg)

        # Initialize milestone from current state (avoid spurious on restart)
        try:
            init_ideas = parse_ideas(cfg["ideas_file"])
            init_counts = _count_statuses(init_ideas, self.results_dir)
            milestone_every = (cfg.get("notifications") or {}).get(
                "milestone_every", 100)
            if milestone_every > 0:
                self._last_milestone = (
                    init_counts.get("COMPLETED", 0) // milestone_every
                ) * milestone_every
                self._hb_completed_count = init_counts.get("COMPLETED", 0)
        except Exception:
            pass

        while self.running:
            self.iteration += 1
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            logger.info("--- Iteration %d [%s] ---", self.iteration, ts)

            # 0a. Early heartbeat — keeps nodes UI alive even when
            #     iterations are slow (large results_dir scans).
            try:
                busy = set(self.active.keys()) | set(self.active_evals.keys())
                free_early = [g for g in self.gpu_ids if g not in busy]
                write_host_heartbeat(self.results_dir,
                                     socket.gethostname(),
                                     self.active, free_early)
            except Exception:
                pass

            # 0b. Auto-upgrade check (rate-limited PyPI + sentinel from other nodes)
            self._check_auto_upgrade()
            self._check_upgrade_sentinel()

            # 0c. Version compatibility check (updates _incompatible_hosts)
            try:
                self._check_cluster_versions()
            except Exception:
                pass

            # 0d. Filesystem health check — pause if FS is not writable
            if not self._health_monitor.check_before_write():
                self._stop_event.wait(self._health_monitor.retry_delay)
                continue

            # 0. Check for filesystem stop/disable signals (multi-machine)
            if self._check_stop_all() or self._check_disabled():
                break

            # 1. Check disk space (only gates launches, never skips reaping)
            disk_ok = check_disk_space(self.results_dir,
                                       cfg.get("min_disk_gb", 0))
            if not disk_ok:
                logger.warning(
                    "Low disk space (< %dGB free). Pausing launches.",
                    cfg["min_disk_gb"])

            # 2. Periodic maintenance (orphans + GC, locked for multi-machine)
            cleanup_cfg = cfg.get("cleanup") or {}
            cleanup_interval = cleanup_cfg.get("interval", 100)
            if cleanup_interval > 0 and self.iteration % cleanup_interval == 0:
                cleanup_lock = self.results_dir / "_cleanup_lock"
                if _fs_lock(cleanup_lock, stale_seconds=300):
                    try:
                        orphan_hours = cfg.get("orphan_timeout_hours", 0)
                        if orphan_hours > 0:
                            cleaned = cleanup_orphans(
                                self.results_dir, orphan_hours)
                            if cleaned:
                                logger.info("Cleaned %d orphaned claims",
                                            cleaned)
                        run_cleanup(self.results_dir, cfg)
                    finally:
                        _fs_unlock(cleanup_lock)
                else:
                    logger.debug("Cleanup lock held by another host, skipping")

            # 3. Check active training processes (with health monitoring)
            finished = []
            if self.active:
                finished = check_active(self.active, self.results_dir,
                                        cfg, self.failure_counts,
                                        self.fix_counts)

            # 3a. Check active eval processes
            eval_finished = []
            if self.active_evals:
                eval_finished = check_active_evals(
                    self.active_evals, self.results_dir, cfg)

            # 3b. Run post-scripts for evals that just completed
            for idea_id, gpu in eval_finished:
                run_post_scripts(idea_id, gpu, self.results_dir, cfg)

            if not self.running:
                break

            # 3c. Auto-upgrade: trigger immediately if pending
            if self._pending_upgrade:
                self._do_auto_upgrade()
                self._pending_upgrade = None

            # 4. Run agent roles (research, documenter, etc.)
            try:
                self._run_all_roles()
            except Exception as e:
                logger.error("Error in _run_all_roles: %s — continuing", e)

            if not self.running:
                break

            # 4b. Mid-iteration heartbeat — keeps nodes alive during long iterations
            try:
                busy = set(self.active.keys()) | set(self.active_evals.keys())
                free_mid = [g for g in self.gpu_ids if g not in busy]
                write_host_heartbeat(self.results_dir,
                                     socket.gethostname(),
                                     self.active, free_mid)
            except Exception:
                pass

            # 5. Sync ideas from ideas.md to idea_lake.db (ingestion)
            raw_ideas = parse_ideas(cfg["ideas_file"])
            if self.lake:
                # Sync new ideas to DB queue
                db_ids = self.lake.get_all_ids()
                ingested_ids = []
                for idea_id, idea in raw_ideas.items():
                    if idea_id not in db_ids:
                        # Clamp priority: "critical" is reserved for
                        # human/API-submitted ideas, not auto-ingested ones.
                        raw_pri = idea.get("priority", "medium")
                        if raw_pri == "critical":
                            raw_pri = "high"
                        raw_text = idea.get("raw", "")
                        def _raw_f(field):
                            m = re.search(rf"\*\*{re.escape(field)}\*\*:\s*(.+)", raw_text)
                            return m.group(1).strip() if m else None
                        self.lake.insert(
                            idea_id, idea["title"], yaml.dump(idea["config"]),
                            raw_text,
                            status="queued",
                            priority=raw_pri,
                            category=_raw_f("Category"),
                            parent=_raw_f("Parent"),
                            hypothesis=_raw_f("Hypothesis"),
                        )
                        ingested_ids.append(idea_id)

                if ingested_ids:
                    logger.info("Ingested %d new ideas from %s to SQLite queue",
                                len(ingested_ids), cfg["ideas_file"])
                    # Consumption: wipe ideas.md after ingestion, keeping header.
                    # Use fs lock to prevent race with concurrent research agent appends.
                    ideas_lock = self.results_dir / ".ideas_md.lock"
                    if _fs_lock(ideas_lock, stale_seconds=60):
                        try:
                            text = Path(cfg["ideas_file"]).read_text(encoding="utf-8")
                            header_match = re.split(r"^## idea-", text, flags=re.MULTILINE)[0]
                            Path(cfg["ideas_file"]).write_text(header_match.strip() + "\n\n",
                                                               encoding="utf-8")
                            logger.info("Consumed %d ideas from %s (wiped file)",
                                        len(ingested_ids), cfg["ideas_file"])
                            # Update pre-snapshots on active roles so the corruption
                            # guard doesn't false-positive on the legitimate wipe.
                            new_size = Path(cfg["ideas_file"]).stat().st_size
                            for rp in self.active_roles.values():
                                rp.ideas_pre_size = new_size
                                rp.ideas_pre_count = 0
                        except Exception as e:
                            logger.warning("Failed to consume ideas.md: %s", e)
                        finally:
                            _fs_unlock(ideas_lock)
                    else:
                        logger.debug("Skipping ideas.md consumption (lock held)")

            # 5a. Expand sweeps and find unclaimed work
            # For sweeps, we still expand in memory for now to keep it simple,
            # but we query the base queue from SQLite.
            sweep_max = cfg.get("sweep", {}).get("max_combos", 20)

            if self.lake:
                # Get the base queue from SQLite
                queue_ideas = {}
                for r in self.lake.get_queue(limit=100):
                    try:
                        cfg_parsed = yaml.safe_load(r["config"]) or {}
                    except yaml.YAMLError:
                        cfg_parsed = {}
                    queue_ideas[r["idea_id"]] = {
                        "title": r["title"],
                        "priority": r["priority"],
                        "config": cfg_parsed,
                        "raw": "",
                    }
                # Expand any sweeps in the queue
                ideas = expand_sweeps(queue_ideas, max_combos=sweep_max)
            else:
                # Legacy fallback
                ideas = expand_sweeps(raw_ideas, max_combos=sweep_max)

            skipped = get_skipped_ideas(
                self.failure_counts,
                cfg.get("max_idea_failures", 0))
            unclaimed = get_unclaimed(ideas, self.results_dir, skipped)
            if unclaimed:
                logger.info("Unclaimed queue (top 5): %s", unclaimed[:5])

            # 6. EVALS FIRST — launch evals before training
            #    Evals are bottleneck (~30min vs 3min training),
            #    so they get priority for free GPUs.
            max_evals = cfg.get("max_concurrent_evals",
                                 len(self.gpu_ids))
            for idea_id, gpu in finished:
                metrics_path = self.results_dir / idea_id / "metrics.json"
                if metrics_path.exists():
                    try:
                        metrics = json.loads(
                            metrics_path.read_text(encoding="utf-8"))
                        if metrics.get("status") == "COMPLETED":
                            if len(self.active_evals) < max_evals:
                                eval_busy = (set(self.active.keys())
                                             | set(self.active_evals.keys()))
                                free_for_eval = [g for g in self.gpu_ids
                                                 if g not in eval_busy]
                                if free_for_eval:
                                    use_gpu = free_for_eval[0]
                                    ep = launch_eval(
                                        idea_id, use_gpu,
                                        self.results_dir, cfg)
                                    if ep is not None:
                                        self.active_evals[use_gpu] = ep
                                    else:
                                        eval_finished.append(
                                            (idea_id, use_gpu))
                                else:
                                    self.pending_evals.append(
                                        (idea_id, gpu))
                                    logger.info(
                                        "Eval deferred for %s (no free GPU)",
                                        idea_id)
                            else:
                                self.pending_evals.append((idea_id, gpu))
                                logger.info(
                                    "Eval deferred for %s (limit %d)",
                                    idea_id, max_evals)
                        else:
                            eval_finished.append((idea_id, gpu))
                    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                        eval_finished.append((idea_id, gpu))
                else:
                    eval_finished.append((idea_id, gpu))

            # 6a. Launch pending evals from previous iterations
            still_pending = []
            for p_idea, p_gpu in self.pending_evals:
                if len(self.active_evals) >= max_evals:
                    still_pending.append((p_idea, p_gpu))
                    continue
                eval_busy = (set(self.active.keys())
                             | set(self.active_evals.keys()))
                free_for_eval = [g for g in self.gpu_ids
                                 if g not in eval_busy]
                if free_for_eval:
                    use_gpu = free_for_eval[0]
                    ep = launch_eval(
                        p_idea, use_gpu, self.results_dir, cfg)
                    if ep is not None:
                        self.active_evals[use_gpu] = ep
                    else:
                        eval_finished.append((p_idea, use_gpu))
                else:
                    still_pending.append((p_idea, p_gpu))
            self.pending_evals = still_pending

            # 6b. Backlog scan: fill remaining eval slots with
            #     completed-but-unevaluated ideas (newest first)
            backlog = []
            if len(self.active_evals) < max_evals:
                eval_output = cfg.get("eval_output",
                                      "eval_report.json")
                pending_ids = {pi for pi, _ in self.pending_evals}
                active_eval_ids = {ep.idea_id
                                   for ep in self.active_evals.values()}
                ckpt_dir = _get_checkpoint_dir(cfg)
                known_ideas = set(ideas.keys())
                backlog = []
                for d in self.results_dir.iterdir():
                    if not d.is_dir() or not d.name.startswith("idea-"):
                        continue
                    iid = d.name
                    if iid in pending_ids or iid in active_eval_ids:
                        continue
                    mpath = d / "metrics.json"
                    rpath = d / eval_output
                    if mpath.exists() and not rpath.exists():
                        # Skip ideas without checkpoints
                        if ckpt_dir and not (
                                ckpt_dir / iid / "best.pt").exists():
                            continue
                        # Only eval ideas we have configs for
                        if iid not in known_ideas:
                            continue
                        try:
                            m = json.loads(
                                mpath.read_text(encoding="utf-8"))
                            if m.get("status") == "COMPLETED":
                                try:
                                    num = int(iid.split("-", 1)[1])
                                except (IndexError, ValueError):
                                    num = 0
                                backlog.append((num, iid))
                        except (json.JSONDecodeError, OSError):
                            pass
                if backlog:
                    backlog.sort(reverse=True)
                    eval_busy = (set(self.active.keys())
                                 | set(self.active_evals.keys()))
                    mem_thresh = cfg.get("gpu_mem_threshold", 2000)
                    free_for_eval = [
                        g for g in self.gpu_ids
                        if g not in eval_busy
                        and (get_gpu_memory_used(g) or 0) <= mem_thresh
                    ]
                    launched_backlog = 0
                    for _, iid in backlog:
                        if (len(self.active_evals) >= max_evals
                                or not free_for_eval):
                            break
                        # Skip if an orphaned eval is already running
                        if _eval_already_running(iid, cfg):
                            continue
                        use_gpu = free_for_eval.pop(0)
                        ep = launch_eval(
                            iid, use_gpu, self.results_dir, cfg)
                        if ep is not None:
                            self.active_evals[use_gpu] = ep
                            launched_backlog += 1
                    if launched_backlog:
                        logger.info(
                            "Launched %d backlog evals (%d remaining)",
                            launched_backlog,
                            len(backlog) - launched_backlog)

            # 7. Launch training on remaining free GPUs
            busy_gpus = set(self.active.keys()) | set(self.active_evals.keys())
            free = [g for g in self.gpu_ids if g not in busy_gpus
                    and (get_gpu_memory_used(g) or 0)
                    <= cfg.get("gpu_mem_threshold", 2000)]

            # Limit concurrent sweep variants per base idea
            max_sweep_concurrent = cfg.get("sweep", {}).get(
                "max_concurrent", 3)
            sweep_counts: Dict[str, int] = {}
            for tp in self.active.values():
                base = tp.idea_id
                if "-ht-" in base:
                    base = base.split("-ht-", 1)[0]
                elif "~" in base:
                    base = base.split("~", 1)[0]
                sweep_counts[base] = sweep_counts.get(base, 0) + 1

            if unclaimed and free and disk_ok:
                for gpu in free:
                    launched = False
                    while unclaimed:
                        idea_id = unclaimed.pop(0)
                        # Enforce per-idea sweep concurrency limit
                        base_id = idea_id
                        if "-ht-" in base_id:
                            base_id = base_id.split("-ht-", 1)[0]
                        elif "~" in base_id:
                            base_id = base_id.split("~", 1)[0]

                        if base_id != idea_id:
                            if sweep_counts.get(base_id, 0) >= max_sweep_concurrent:
                                continue
                        if not claim(idea_id, self.results_dir, gpu):
                            continue
                        # Write sweep config for sub-runs
                        if ideas.get(idea_id, {}).get("_sweep_parent"):
                            atomic_write(
                                self.results_dir / idea_id / "sweep_config.yaml",
                                yaml.dump(ideas[idea_id]["config"],
                                          default_flow_style=False))
                        if not run_pre_script(idea_id, gpu, cfg):
                            logger.warning(
                                "Pre-script failed for %s, marking FAILED",
                                idea_id)
                            error_msg = "Pre-script failed"
                            if _try_executor_fix(idea_id, error_msg,
                                                 self.results_dir, cfg,
                                                 self.fix_counts):
                                _reset_idea_for_retry(
                                    self.results_dir / idea_id)
                                if run_pre_script(idea_id, gpu, cfg):
                                    pass  # fixed — fall through to launch
                                else:
                                    _write_failure(
                                        self.results_dir / idea_id,
                                        "Pre-script failed after fix")
                                    _record_failure(
                                        self.failure_counts, idea_id)
                                    continue
                            else:
                                _write_failure(
                                    self.results_dir / idea_id, error_msg)
                                _record_failure(
                                    self.failure_counts, idea_id)
                                continue
                        logger.info("Launching %s on GPU %d: %s",
                                    idea_id, gpu,
                                    ideas[idea_id]["title"][:50])
                        try:
                            tp = launch(idea_id, gpu, self.results_dir, cfg)
                        except Exception as e:
                            logger.error("Failed to launch %s on GPU %d: %s",
                                         idea_id, gpu, e)
                            error_msg = f"Launch error: {e}"
                            if _try_executor_fix(idea_id, error_msg,
                                                 self.results_dir, cfg,
                                                 self.fix_counts):
                                _reset_idea_for_retry(
                                    self.results_dir / idea_id)
                                try:
                                    tp = launch(idea_id, gpu,
                                                self.results_dir, cfg)
                                except Exception as e2:
                                    logger.error(
                                        "[FIX-RETRY] %s relaunch failed: %s",
                                        idea_id, e2)
                                    _write_failure(
                                        self.results_dir / idea_id,
                                        f"Launch error after fix: {e2}")
                                    _record_failure(
                                        self.failure_counts, idea_id)
                                    continue
                            else:
                                _write_failure(self.results_dir / idea_id,
                                               error_msg)
                                _record_failure(self.failure_counts, idea_id)
                                continue
                        self.active[gpu] = tp
                        base_id = idea_id
                        if "-ht-" in base_id:
                            base_id = base_id.split("-ht-", 1)[0]
                        elif "~" in base_id:
                            base_id = base_id.split("~", 1)[0]

                        if base_id != idea_id:
                            sweep_counts[base_id] = sweep_counts.get(base_id, 0) + 1
                        launched = True
                        break
                    if not launched:
                        break
            elif not unclaimed:
                if not self.active and not self.active_evals:
                    logger.info("All ideas completed or skipped!")
                    if not self.once:
                        logger.info("Waiting for new ideas...")
                else:
                    logger.info("No unclaimed ideas. %d training, %d eval.",
                                len(self.active), len(self.active_evals))
            else:
                logger.info("%d ideas queued, no free GPUs (%d training, "
                            "%d eval)",
                            len(unclaimed), len(self.active),
                            len(self.active_evals))

            # Circuit Breaker: if too many ideas exceeded max retries, stop the farm
            max_fail = cfg.get("max_idea_failures", 0)
            if max_fail > 0 and len(self.failure_counts) > 5:
                exhausted = [fid for fid, count in self.failure_counts.items()
                             if count >= max_fail]
                if len(exhausted) > 10:
                    logger.error("CIRCUIT BREAKER: %d ideas exhausted %d retries. Stopping.",
                                 len(exhausted), max_fail)
                    (self.results_dir / ".orze_stop_all").touch()

            # 8. Update report
            completed_rows = update_report(self.results_dir, ideas, cfg, lake=self.lake)

            # 9. Write heartbeat + status.json
            write_host_heartbeat(self.results_dir, socket.gethostname(), self.active, free)
            counts = _count_statuses(ideas, self.results_dir)

            # 8a. Notifications (fires for eval-finished ideas, metrics available)
            self._process_notifications(
                eval_finished, completed_rows or [], ideas, counts)

            # 8b. Heartbeat (rate-controlled, default 1800s = 30 min)
            heartbeat_interval = (cfg.get("notifications") or {}).get(
                "heartbeat_interval", 1800)
            if heartbeat_interval > 0:
                now_hb = time.time()
                if now_hb - self._last_heartbeat >= heartbeat_interval:
                    uptime_s = int(now_hb - self._start_time)
                    h, rem = divmod(uptime_s, 3600)
                    m = rem // 60
                    uptime_str = f"{h}h{m:02d}m" if h else f"{m}m"
                    busy_set = set(self.active.keys()) | set(self.active_evals.keys())
                    n_free = len([g for g in self.gpu_ids if g not in busy_set])
                    notify("heartbeat", {
                        "host": socket.gethostname(),
                        "iteration": self.iteration,
                        "uptime": uptime_str,
                        "training": len(self.active),
                        "eval": len(self.active_evals),
                        "free": n_free,
                        "completed": counts.get("COMPLETED", 0),
                        "queued": counts.get("QUEUED", 0),
                        "failed": counts.get("FAILED", 0),
                        "eval_backlog": len(backlog),
                        "rate": (f"{counts.get('COMPLETED', 0) - self._hb_completed_count}"
                                 f" since last heartbeat"),
                    }, cfg)
                    self._last_heartbeat = now_hb
                    self._hb_completed_count = counts.get("COMPLETED", 0)

            # 8c. Milestone (every N completions, default 100)
            milestone_every = (cfg.get("notifications") or {}).get(
                "milestone_every", 100)
            if milestone_every > 0:
                completed_now = counts.get("COMPLETED", 0)
                curr_milestone = (completed_now // milestone_every) * milestone_every
                if curr_milestone > self._last_milestone and curr_milestone > 0:
                    notify("milestone", {"count": curr_milestone}, cfg)
                    self._last_milestone = curr_milestone

            # 8d. Disk warning (at most once per 30 min)
            if not disk_ok and time.time() - self._last_disk_warning > 1800:
                try:
                    usage = shutil.disk_usage(self.results_dir)
                    free_gb = usage.free / (1024 ** 3)
                except Exception:
                    free_gb = "?"
                notify("disk_warning", {
                    "host": socket.gethostname(),
                    "free_gb": f"{free_gb:.1f}" if isinstance(free_gb, float) else free_gb,
                }, cfg)
                self._last_disk_warning = time.time()

            top_results = []
            if completed_rows:
                primary = cfg["report"].get("primary_metric",
                                            "test_accuracy")
                for r in completed_rows[:10]:
                    top_results.append({
                        "idea_id": r["id"],
                        "title": r["title"][:60],
                        primary: r.get("primary_val"),
                    })

            write_status_json(
                self.results_dir, self.iteration, self.active, free,
                len(unclaimed), counts.get("COMPLETED", 0),
                counts.get("FAILED", 0), len(skipped), top_results, cfg,
                role_states=self.role_states,
            )

            # 9b. Admin cache (pre-aggregated nodes/queue/alerts)
            try:
                write_admin_cache(self.results_dir, ideas, cfg)
            except Exception as e:
                logger.warning("Admin cache write failed: %s", e)

            # 10. Save state
            save_state(self.results_dir, {
                "iteration": self.iteration,
                "failure_counts": self.failure_counts,
                "fix_counts": self.fix_counts,
                "roles": self.role_states,
                "best_idea_id": self._best_idea_id,
            })

            if self.once:
                all_once_finished = []
                # Wait for active training
                if self.active:
                    logger.info("--once mode: waiting for active training...")
                    while self.active:
                        time.sleep(5)
                        once_finished = check_active(
                            self.active, self.results_dir,
                            cfg, self.failure_counts,
                            self.fix_counts)
                        for idea_id, gpu in once_finished:
                            m_path = self.results_dir / idea_id / "metrics.json"
                            if m_path.exists():
                                try:
                                    m = json.loads(m_path.read_text(encoding="utf-8"))
                                    if m.get("status") == "COMPLETED":
                                        run_eval(idea_id, gpu,
                                                 self.results_dir, cfg)
                                        run_post_scripts(idea_id, gpu,
                                                         self.results_dir, cfg)
                                except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                                    pass
                            all_once_finished.append((idea_id, gpu))
                # Wait for active evals (launched this iteration or earlier)
                if self.active_evals:
                    logger.info("--once mode: waiting for %d active evals...",
                                len(self.active_evals))
                    while self.active_evals:
                        time.sleep(5)
                        ef = check_active_evals(
                            self.active_evals, self.results_dir, cfg)
                        for idea_id, gpu in ef:
                            run_post_scripts(
                                idea_id, gpu, self.results_dir, cfg)
                            all_once_finished.append((idea_id, gpu))
                if all_once_finished:
                    ideas = parse_ideas(cfg["ideas_file"])
                    once_rows = update_report(self.results_dir, ideas, cfg)
                    once_counts = _count_statuses(ideas, self.results_dir)
                    self._process_notifications(
                        all_once_finished, once_rows or [], ideas,
                        once_counts)
                    save_state(self.results_dir, {
                        "iteration": self.iteration,
                        "failure_counts": self.failure_counts,
                        "fix_counts": self.fix_counts,
                        "roles": self.role_states,
                        "best_idea_id": self._best_idea_id,
                    })
                logger.info("Done.")
                break

            # Interruptible sleep — write heartbeat every 60s while waiting
            poll_remaining = cfg["poll"]
            while poll_remaining > 0 and not self._stop_event.is_set():
                tick = min(poll_remaining, 60)
                self._stop_event.wait(tick)
                poll_remaining -= tick
                if not self._stop_event.is_set():
                    try:
                        busy = set(self.active.keys()) | set(self.active_evals.keys())
                        free = [g for g in self.gpu_ids if g not in busy]
                        write_host_heartbeat(self.results_dir,
                                             socket.gethostname(),
                                             self.active, free)
                    except Exception:
                        pass
            if self._stop_event.is_set():
                break

        # Main loop exited (signal received or --once finished)
        if self.active or self.active_evals or self.active_roles:
            self._graceful_shutdown(
                kill_all=getattr(self, '_stop_kill_all', False))
        else:
            # Nothing running, just save state and clean up
            save_state(self.results_dir, {
                "iteration": self.iteration,
                "failure_counts": self.failure_counts,
                "fix_counts": self.fix_counts,
                "roles": self.role_states,
                "best_idea_id": self._best_idea_id,
            })
            if self.lake:
                try:
                    self.lake.close()
                except Exception:
                    pass
            self._remove_pid_file()
        logger.info("Exited after %d iterations.", self.iteration)
