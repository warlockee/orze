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
    _kill_pg,
    run_pre_script,
)
from orze.engine.scheduler import (
    claim, get_unclaimed, cleanup_orphans, _count_statuses, run_cleanup,
)
from orze.engine.launcher import (
    launch, check_active, _get_checkpoint_dir, _write_failure,
)
from orze.engine.evaluator import (
    launch_eval, check_active_evals, run_eval, run_post_scripts,
)
from orze.engine.health import check_disk_space
from orze.engine.failure import (
    _record_failure, get_skipped_ideas, _try_executor_fix, _reset_idea_for_retry,
)
from orze.core.fs import _fs_lock, _fs_unlock, atomic_write
from orze.core.ideas import parse_ideas, expand_sweeps
from orze.core.config import _validate_config
from orze.reporting.state import (
    load_state, save_state, write_host_heartbeat,
    write_status_json,
)
from orze.reporting.leaderboard import (
    update_report, write_admin_cache,
)
from orze.reporting.notifications import notify
from orze.hardware.gpu import get_gpu_memory_used, _eval_already_running
from orze.engine.config_dedup import hash_config, load_hashes, save_hash, rebuild_hashes
from orze.engine.cluster import (
    check_cluster_versions as _cluster_check_versions,
    build_machine_status, check_stop_all, check_disabled, kill_orphans,
)
from orze.engine.upgrade import UpgradeManager
from orze.engine.reporter import NotificationProcessor
from orze.engine.retrospection import run_retrospection
from orze.engine.role_runner import (
    RoleContext, run_role_step, run_all_roles as _run_all_roles_impl,
    run_role_once as _run_role_once_impl, build_claude_cmd, build_research_cmd,
)
from orze.engine.lifecycle import (
    startup_checks, reconcile_stale_running, print_startup_summary,
    graceful_shutdown, atexit_cleanup, write_shutdown_heartbeat,
    write_pid_file, remove_pid_file,
)
from orze.engine.phases import OrzePhaseMixin

logger = logging.getLogger("orze")


class Orze(OrzePhaseMixin):
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
        self._start_time: float = time.time()
        self._last_heartbeat: float = 0.0
        self._hb_completed_count: int = 0  # for heartbeat rate calc
        self._last_milestone: int = 0      # last milestone boundary hit
        self._last_disk_warning: float = 0.0

        self._pending_upgrade: Optional[str] = None

        # Retrospection state
        self._retro_last_count = state.get("retro_last_count", 0)

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

        # Managers (extracted from this file for LOD compliance)
        self._upgrade_mgr = UpgradeManager(self.results_dir, cfg)
        self._reporter = NotificationProcessor(self.results_dir, cfg, lake=self.lake)
        self._reporter.load_state(state)

        # Event used to interrupt poll sleep instantly on shutdown signal
        self._stop_event = threading.Event()

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

        # Safety net: atexit kills all children even if signal handler
        # can't run (e.g. unhandled exception, sys.exit from 2nd signal)
        atexit.register(self._atexit_cleanup)

    def _atexit_cleanup(self):
        atexit_cleanup(self.active, self.active_evals, self.active_roles)

    def _build_state_dict(self):
        """Build state dict for persistence, merging reporter state."""
        reporter_state = self._reporter.get_state()
        return {
            "iteration": self.iteration,
            "failure_counts": self.failure_counts,
            "fix_counts": self.fix_counts,
            "roles": self.role_states,
            "retro_last_count": self._retro_last_count,
            **reporter_state,
        }

    # -- Config deduplication helpers --

    def _config_override_hash(self, config: dict) -> str:
        """Hash an idea's config overrides for dedup detection."""
        return hash_config(config)

    def _load_config_hashes(self) -> dict:
        """Load the config hash -> idea_id mapping."""
        return load_hashes(self.results_dir)

    def _save_config_hash(self, idea_id: str, config: dict):
        """Store a completed idea's config hash."""
        save_hash(self.results_dir, idea_id, config)

    def _rebuild_config_hashes(self):
        """Rebuild config hash cache from existing completed ideas' resolved configs."""
        rebuild_hashes(self.results_dir)

    def _startup_checks(self):
        self._health_monitor = startup_checks(
            self.results_dir, self.cfg, self._hostname, self._instance_uuid)

    def _reconcile_stale_running(self):
        reconcile_stale_running(self.cfg)

    def _print_startup_summary(self):
        print_startup_summary(self.cfg)

    def _shutdown(self, signum, frame):
        """Signal handler — sets flag and wakes poll sleep immediately."""
        if not self.running:
            # Second signal = force exit
            logger.warning("Forced exit (second signal).")
            sys.exit(1)
        logger.info("Received signal %d, shutting down...", signum)
        self.running = False
        self._stop_event.set()  # wake from poll sleep instantly

    def _graceful_shutdown(self, kill_all=False):
        graceful_shutdown(
            self.results_dir, self.cfg, self.active, self.active_evals,
            self.active_roles, self.iteration, self._build_state_dict(),
            self.lake, self._hostname, self._instance_uuid, kill_all=kill_all)

    def _write_shutdown_heartbeat(self):
        write_shutdown_heartbeat(self.results_dir, self._hostname,
                                 self._instance_uuid, self.active)

    def _check_cluster_versions(self):
        """Check version compatibility with other nodes. Updates _incompatible_hosts."""
        heartbeats, self._incompatible_hosts = _cluster_check_versions(self.results_dir)
        return heartbeats

    def _build_machine_status(self) -> list:
        """Build machine status from heartbeats for report notifications."""
        return build_machine_status(self.results_dir)

    def _process_notifications(self, finished: list,
                               completed_rows: list, ideas: dict,
                               counts: dict):
        """Fire notifications for finished experiments and new bests. Never raises."""
        self._reporter.process(
            finished, completed_rows, ideas, counts,
            len(self.active),
            save_config_hash_fn=lambda idea_id, config: save_hash(self.results_dir, idea_id, config),
            build_machine_status_fn=lambda: build_machine_status(self.results_dir),
        )

    def _run_role_step(self, role_name, role_cfg):
        ctx = self._role_context()
        run_role_step(role_name, role_cfg, ctx)

    def _run_role_once(self, role_name):
        ctx = self._role_context()
        _run_role_once_impl(role_name, ctx)

    def _run_all_roles(self):
        ctx = self._role_context()
        _run_all_roles_impl(ctx)

    def _build_claude_cmd(self, research_cfg, template_vars):
        return build_claude_cmd(research_cfg, template_vars)

    def _build_research_cmd(self, role_cfg, template_vars):
        return build_research_cmd(role_cfg, template_vars, self.cfg)

    def _role_context(self):
        return RoleContext(
            cfg=self.cfg,
            results_dir=self.results_dir,
            gpu_ids=self.gpu_ids,
            active_roles=self.active_roles,
            role_states=self.role_states,
            failure_counts=self.failure_counts,
            fix_counts=self.fix_counts,
            iteration=self.iteration,
        )

    def _run_retrospection(self, completed_count):
        """Run user-provided retrospection script periodically."""
        self._retro_last_count = run_retrospection(
            self.results_dir, self.cfg, completed_count, self._retro_last_count)

    def _kill_orphans(self):
        """Kill orphaned train/eval processes from a previous Orze instance."""
        kill_orphans(self.results_dir, self.cfg)

    def _write_pid_file(self):
        self._pid_file = write_pid_file(self.results_dir)

    def _remove_pid_file(self):
        remove_pid_file(getattr(self, '_pid_file', None), self.results_dir)

    def _check_stop_all(self):
        """Check for filesystem-based stop signal (.orze_stop_all)."""
        should_stop, kill_all = check_stop_all(self.results_dir)
        if should_stop:
            self._stop_kill_all = kill_all
            self.running = False
        return should_stop

    def _check_disabled(self):
        """Check for persistent disable flag (.orze_disabled)."""
        return check_disabled(self.results_dir)

    def _check_auto_upgrade(self):
        """Check PyPI for a newer orze version. Rate-limited. Never raises."""
        self._upgrade_mgr.check_pypi()
        self._pending_upgrade = self._upgrade_mgr.pending

    def _check_upgrade_sentinel(self):
        """Check if another node wrote .orze_upgrade sentinel.
        If the target version is newer, restart via os.execv (pip already done)."""
        self._upgrade_mgr.check_sentinel()
        if self._upgrade_mgr.pending:
            self._pending_upgrade = self._upgrade_mgr.pending
            self._do_auto_upgrade()

    def _do_auto_upgrade(self):
        """Install pending upgrade, kill everything, and restart via os.execv."""
        self._upgrade_mgr.pending = self._pending_upgrade
        self._upgrade_mgr.do_upgrade(self._kill_and_save, self._remove_pid_file)
        self._pending_upgrade = None

    def _kill_and_save(self):
        """Kill all children, save state, close lake. Used as upgrade callback."""
        for gpu, tp in self.active.items():
            _kill_pg(tp.process, signal.SIGTERM)
        for role_name, rp in self.active_roles.items():
            _kill_pg(rp.process, signal.SIGTERM)
        deadline = time.time() + 10
        for gpu, tp in list(self.active.items()):
            try:
                tp.process.wait(timeout=max(1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                _kill_pg(tp.process, signal.SIGKILL)
            tp.close_log()
        for gpu, ep in list(self.active_evals.items()):
            ep.close_log()
        for role_name, rp in list(self.active_roles.items()):
            try:
                rp.process.wait(timeout=max(1, deadline - time.time()))
            except subprocess.TimeoutExpired:
                _kill_pg(rp.process, signal.SIGKILL)
            rp.close_log()
            _fs_unlock(rp.lock_dir)
        try:
            save_state(self.results_dir, self._build_state_dict())
        except Exception as e:
            logger.warning("Auto-upgrade: state save failed: %s", e)
        if self.lake:
            try:
                self.lake.close()
            except Exception:
                pass

    # Phase methods (_sync_ideas, _launch_evals, _launch_training,
    # _report_and_notify) live in OrzePhaseMixin (phases.py)

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

        # Full reconcile at startup: clear ALL stale queued ideas at once
        if self.lake:
            try:
                n = self.lake.reconcile_statuses(str(self.results_dir))
                if n:
                    logger.info("Startup reconcile: updated %d stale ideas", n)
            except Exception as e:
                logger.warning("Startup reconcile failed: %s", e)

        # Rebuild config dedup hash cache from completed ideas
        try:
            self._rebuild_config_hashes()
        except Exception as e:
            logger.warning("Config hash cache rebuild failed: %s", e)

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
                                self.results_dir, orphan_hours,
                                lake=self.lake)
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

            # 5. Sync ideas + expand sweeps + build unclaimed queue
            ideas, unclaimed, skipped, raw_ideas = self._sync_ideas(cfg)

            # 6. Launch evals (finished training, pending, backlog)
            eval_finished, backlog = self._launch_evals(
                finished, eval_finished, ideas)

            # 7. Launch training on free GPUs + circuit breaker
            free = self._launch_training(unclaimed, disk_ok, ideas)

            # 8. Update report
            completed_rows = update_report(self.results_dir, ideas, cfg, lake=self.lake)

            # 9-10. Report, notify, heartbeat, status.json, save state
            write_host_heartbeat(self.results_dir, socket.gethostname(), self.active, free)
            counts = _count_statuses(ideas, self.results_dir)
            self._report_and_notify(
                completed_rows, ideas, counts, eval_finished,
                free, unclaimed, skipped, disk_ok, backlog)

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
                    save_state(self.results_dir, self._build_state_dict())
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
            save_state(self.results_dir, self._build_state_dict())
            if self.lake:
                try:
                    self.lake.close()
                except Exception:
                    pass
            self._remove_pid_file()
        logger.info("Exited after %d iterations.", self.iteration)
