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
from orze.engine.leader import (
    try_acquire as _leader_try_acquire,
    read_current_leader as _leader_read_current,
    should_skip_role_as_follower as _leader_skip_role,
)
from orze.extensions import get_extension as _get_ext

_role_mod = _get_ext("role_runner")
if _role_mod:
    RoleContext = _role_mod.RoleContext
    run_role_step = _role_mod.run_role_step
    _run_all_roles_impl = _role_mod.run_all_roles
    _run_role_once_impl = _role_mod.run_role_once
    build_claude_cmd = _role_mod.build_claude_cmd
    build_research_cmd = _role_mod.build_research_cmd
else:
    # No pro / no built-in agents: stub everything
    RoleContext = None
    run_role_step = None
    _run_all_roles_impl = None
    _run_role_once_impl = None
    build_claude_cmd = None
    build_research_cmd = None
from orze.engine.lifecycle import (
    startup_checks, reconcile_stale_running, print_startup_summary,
    graceful_shutdown, atexit_cleanup, write_shutdown_heartbeat,
    write_pid_file, remove_pid_file,
)
from orze.engine.phases import OrzePhaseMixin

logger = logging.getLogger("orze")

_roles_unavailable_warned = False


class Orze(OrzePhaseMixin):
    def __init__(self, gpu_ids: List[int], cfg: dict, once: bool = False):
        self.gpu_ids = gpu_ids
        self.cfg = cfg
        self.once = once
        self.results_dir = Path(cfg["results_dir"])
        # GPU scheduler: "auto" mode starts exclusive, then upgrades to VRAM
        # packing after the first job completes if it used <30% of GPU VRAM.
        from orze.engine.gpu_slots import GpuSlotManager
        sched_cfg = cfg.get("gpu_scheduling", {})
        mode = sched_cfg.get("mode", "auto")
        self._auto_gpu_mode = (mode == "auto")
        if mode == "auto":
            mode = "exclusive"  # start safe, upgrade later
            logger.info("GPU scheduling: auto mode — starting exclusive, "
                        "will upgrade to VRAM packing if jobs are small")
        self.slot_mgr = GpuSlotManager(
            gpu_ids,
            mode=mode,
            max_vram_pct=sched_cfg.get("max_vram_pct", 90),
            min_free_vram_mib=sched_cfg.get("min_free_vram_mib", 1000),
            max_jobs_per_gpu=sched_cfg.get("max_jobs_per_gpu", 200),
            max_load_per_cpu=sched_cfg.get("max_load_per_cpu", 2.0),
            min_free_ram_gb=sched_cfg.get("min_free_ram_gb", 16.0),
            slots_per_gpu=sched_cfg.get("slots_per_gpu", 1),  # legacy compat
        )
        self.active = self.slot_mgr  # dict-compatible drop-in
        self.active_evals: Dict[int, EvalProcess] = {}
        self.active_roles: Dict[str, RoleProcess] = {}
        self.pending_evals: list = []
        self.running = True
        self._hostname = socket.gethostname()
        self._instance_uuid = uuid.uuid4().hex[:12]
        self._incompatible_hosts: set = set()

        # Leader election state: set by _acquire_leadership on run().
        # None = not yet attempted. Falsy handle = we are a follower.
        self._leader_handle = None
        self._follower_last_log: float = 0.0

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
        self._last_heartbeat: float = self._start_time
        self._hb_completed_count: int = 0  # for heartbeat rate calc
        self._last_milestone: int = 0      # last milestone boundary hit
        self._last_disk_warning: float = 0.0

        self._pending_upgrade: Optional[str] = None

        # Retrospection state
        self._retro_last_count = state.get("retro_last_count", 0)
        self._retro_state: dict = state.get("retro_state", {})

        # Initialize Idea Lake for archival
        try:
            from orze.idea_lake import IdeaLake
            lake_path = Path(cfg["idea_lake_db"])

            # Migrate old location (next to ideas.md) to new location (results_dir)
            old_lake = Path(cfg.get("ideas_file", "ideas.md")).parent / "idea_lake.db"
            if old_lake != lake_path and old_lake.exists() and not lake_path.exists():
                lake_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(str(old_lake), str(lake_path))
                logger.info("Migrated idea_lake.db: %s -> %s", old_lake, lake_path)

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
            self.lake = None
            # Check if results already exist — if so, this is a serious
            # degradation (not a first run), so escalate to ERROR + notify.
            res_dir = Path(cfg.get("results_dir", "results"))
            has_results = (res_dir.exists() and
                any(d.is_dir() and d.name.startswith("idea-")
                    for d in res_dir.iterdir())
                if res_dir.exists() else False)
            if has_results:
                logger.error("Idea Lake init FAILED with existing results — "
                             "archival disabled: %s", exc)
                notify("idea_lake_failure", {
                    "message": f"IdeaLake init failed: {exc}. "
                               f"Results exist but archival is disabled.",
                }, cfg)
            else:
                logger.warning("Idea Lake not available: %s", exc)

        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Managers (extracted from this file for LOD compliance)
        self._upgrade_mgr = UpgradeManager(self.results_dir, cfg)
        self._reporter = NotificationProcessor(self.results_dir, cfg, lake=self.lake)
        self._reporter.load_state(state)
        # F4: rebuild best_idea_id / completions_since_best from lake if
        # state lost them (e.g. upgrade reset). Idempotent fill-only.
        try:
            from orze.engine.rebuild_state import rebuild_best_from_lake
            if state.get("best_idea_id") is None and self.lake is not None:
                primary = cfg.get("report", {}).get("primary_metric",
                                                    "test_accuracy")
                best_id, since = rebuild_best_from_lake(self.lake, primary)
                if best_id is not None:
                    self._reporter._best_idea_id = best_id
                    self._reporter._completions_since_best = since
                    logger.info("Rebuilt best_idea_id=%s "
                                "completions_since_best=%d from idea_lake "
                                "(metric=%s)", best_id, since, primary)
        except Exception as e:
            logger.debug("rebuild_best_from_lake failed on startup: %s", e)

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
            "retro_state": self._retro_state,
            **reporter_state,
        }

    # -- Config deduplication helpers --

    def _config_override_hash(self, config: dict) -> str:
        """Hash an idea's config overrides for dedup detection."""
        return hash_config(config)

    def _load_config_hashes(self) -> dict:
        """Load the config hash -> idea_id mapping."""
        return load_hashes(self.results_dir, self.cfg)

    def _save_config_hash(self, idea_id: str, config: dict):
        """Store a completed idea's config hash."""
        save_hash(self.results_dir, idea_id, config, self.cfg)

    def _rebuild_config_hashes(self):
        """Rebuild config hash cache from existing completed ideas' resolved configs."""
        rebuild_hashes(self.results_dir, self.cfg)

    def _startup_checks(self):
        self._health_monitor = startup_checks(
            self.results_dir, self.cfg, self._hostname, self._instance_uuid)
        # F3: migrate any stray ideas.md.corrupt.* files from cwd into
        # .orze/backups/corrupt/ and prune to last 5.
        try:
            from orze.engine.roles import cleanup_stale_corrupt_files
            ideas_path = Path(self.cfg.get("ideas_file", "ideas.md"))
            moved = cleanup_stale_corrupt_files(ideas_path, self.cfg)
            if moved:
                logger.info(
                    "Archived %d stray ideas.md.corrupt.* files to "
                    ".orze/backups/corrupt/ (kept last 5)", moved)
        except Exception as e:
            logger.debug("cleanup_stale_corrupt_files: %s", e)
        # F6: relocate zero-byte stale DB files at CWD out of the way.
        try:
            self._cleanup_stale_root_dbs()
        except Exception as e:
            logger.debug("cleanup_stale_root_dbs: %s", e)

    def _cleanup_stale_root_dbs(self):
        """Move zero-byte ``*.db`` files at CWD that collide with canonical
        DB names into ``<results>/_stale/``. They confuse sqlite-connect
        callers into silently creating empty tables."""
        from orze.engine.stale_dbs import relocate_zero_byte_dbs
        moved = relocate_zero_byte_dbs(Path.cwd(), self.results_dir / "_stale")
        for src, dest in moved:
            logger.warning(
                "Moved stale 0-byte %s at CWD to %s (canonical DB is at "
                "results/idea_lake.db)", src, dest)

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
        if run_role_step is None:
            return  # no agent support without pro
        ctx = self._role_context()
        run_role_step(role_name, role_cfg, ctx)

    def _run_role_once(self, role_name):
        if _run_role_once_impl is None:
            logger.info("Role '%s' requires orze-pro. Install with: pip install orze-pro", role_name)
            return
        ctx = self._role_context()
        _run_role_once_impl(role_name, ctx)

    def _acquire_leadership(self) -> None:
        """Attempt to acquire the cross-host leader lock once per tick.

        We re-try on every tick so that when the current leader dies,
        this follower becomes the new leader automatically. If we were
        already leader, just refresh the heartbeat.
        """
        if self._leader_handle is not None:
            self._leader_handle.heartbeat()
            return
        handle = _leader_try_acquire(self.results_dir)
        if handle is not None:
            self._leader_handle = handle
            logger.info("leader acquired host=%s pid=%d",
                        handle.host, handle.pid)

    def _is_leader(self) -> bool:
        return self._leader_handle is not None

    def _log_follower_status_rate_limited(self, role_name: str) -> None:
        """Log ``follower mode: skipping role X`` at most once per minute."""
        now = time.time()
        if now - self._follower_last_log < 60.0:
            return
        self._follower_last_log = now
        current = _leader_read_current(self.results_dir) or {}
        who = f"{current.get('host', '?')}:{current.get('pid', '?')}"
        logger.info("follower mode: skipping role %s (leader=%s)",
                    role_name, who)

    def _run_all_roles(self):
        global _roles_unavailable_warned
        if _run_all_roles_impl is None:
            if not _roles_unavailable_warned and self.cfg.get("roles"):
                logger.warning(
                    "Roles configured but agent support unavailable "
                    "(orze-pro not installed or not licensed). Roles will not run."
                )
                _roles_unavailable_warned = True
            return
        # Multi-host coordination: only the leader drives LLM-role cycles.
        # Followers still run experiment execution and metric harvesting.
        self._acquire_leadership()
        if not self._is_leader():
            self._log_follower_status_rate_limited("<all llm roles>")
            return
        ctx = self._role_context()
        _run_all_roles_impl(ctx)

    def _run_role_step_leader_gated(self, role_name, role_cfg):
        """Like _run_role_step but gates LLM-token roles on leadership."""
        if _leader_skip_role(role_name):
            self._acquire_leadership()
            if not self._is_leader():
                self._log_follower_status_rate_limited(role_name)
                return
        self._run_role_step(role_name, role_cfg)

    def _build_claude_cmd(self, research_cfg, template_vars):
        if build_claude_cmd is None:
            return None
        return build_claude_cmd(research_cfg, template_vars)

    def _build_research_cmd(self, role_cfg, template_vars):
        return build_research_cmd(role_cfg, template_vars, self.cfg)

    def _role_context(self):
        if RoleContext is None:
            return None
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

    # Keys that are safe to hot-reload without restart
    _HOT_RELOAD_KEYS = {
        "retrospection", "stall_minutes", "timeout", "poll", "roles",
        "max_idea_failures", "max_fix_attempts", "notifications",
        "plateau_threshold", "orphan_timeout_hours", "gpu_mem_threshold",
        "gpu_scheduling", "min_disk_gb",
    }

    def _hot_reload_config(self):
        """Reload orze.yaml and update safe-to-change config keys.

        For 'roles', merges disk config with runtime-added roles (e.g.
        auto-enabled thinker/data_analyst) instead of replacing outright.
        """
        cfg_path = self.cfg.get("_config_path")
        if not cfg_path or not Path(cfg_path).exists():
            return
        try:
            raw = yaml.safe_load(Path(cfg_path).read_text(encoding="utf-8")) or {}
            changed = []
            for key in self._HOT_RELOAD_KEYS:
                if key not in raw:
                    continue
                if raw[key] == self.cfg.get(key):
                    continue
                if key == "roles":
                    # Merge: update existing roles from disk, but preserve
                    # runtime-added roles (auto-enabled by orze-pro) that
                    # are not in the on-disk config.
                    disk_roles = raw[key] or {}
                    live_roles = self.cfg.get("roles") or {}
                    merged = dict(live_roles)  # keep runtime-added roles
                    merged.update(disk_roles)  # disk config wins for shared keys
                    if merged != live_roles:
                        self.cfg[key] = merged
                        changed.append(key)
                else:
                    self.cfg[key] = raw[key]
                    changed.append(key)
            if changed:
                logger.info("Config hot-reloaded: %s", ", ".join(changed))
        except Exception as e:
            logger.warning("Config hot-reload failed: %s", e)

    def _run_retrospection(self, completed_count):
        """Run retrospection with dispatch to evolution roles."""
        self._retro_last_count = run_retrospection(
            self.results_dir, self.cfg, completed_count, self._retro_last_count,
            retro_state=self._retro_state)

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
        """Check PyPI for a newer orze version. Rate-limited.

        Never auto-upgrades. Only logs a warning and sends a one-time
        notification. The user triggers the upgrade manually via
        `orze --upgrade` or `orze restart`.
        """
        self._upgrade_mgr.check_pypi()
        available = self._upgrade_mgr.pending
        if available and available != self._pending_upgrade:
            # First time seeing this version — warn once
            self._pending_upgrade = available
            logger.warning("New orze v%s available (current v%s). "
                           "Run 'orze --upgrade' to install.",
                           available, __version__)
            try:
                notify("upgrade_available", {
                    "host": socket.gethostname(),
                    "current_version": __version__,
                    "available_version": available,
                    "message": f"orze v{available} available (current v{__version__}). "
                               f"Run 'orze --upgrade' to install.",
                }, self._cfg)
            except Exception:
                pass

    def _check_upgrade_sentinel(self):
        """Check if another node wrote .orze_upgrade sentinel.

        If another node already upgraded (pip install done), this node
        should restart to pick up the new code. This is the only case
        where automatic restart is safe — pip is already done, we just
        need to exec the new binary.
        """
        self._upgrade_mgr.check_sentinel()
        if self._upgrade_mgr.pending:
            self._pending_upgrade = self._upgrade_mgr.pending
            self._do_auto_upgrade()

    def _do_auto_upgrade(self):
        """Restart to pick up an already-installed upgrade (sentinel-triggered only)."""
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

        # Log pro status
        from orze.extensions import has_pro, pro_version
        if has_pro() and _run_all_roles_impl is not None:
            logger.info("orze-pro %s detected — autopilot features enabled", pro_version())
        elif has_pro() and _run_all_roles_impl is None:
            logger.warning("orze-pro installed but not activated — check license with ORZE_PRO_KEY")
        elif _role_mod is not None:
            logger.info("Using built-in agent modules (install orze-pro to upgrade)")
        else:
            roles = cfg.get("roles", {})
            if roles:
                logger.warning(
                    "Roles configured (%s) but no agent support available. "
                    "Install orze-pro for autonomous research agents.",
                    ", ".join(roles.keys()))

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
            if rmode == "claude":
                skills = rcfg.get("skills") or []
                rtarget = f"{len(skills)} skill(s)" if skills else None
            elif rmode == "research":
                skills = rcfg.get("skills") or []
                rtarget = f"{rcfg.get('backend', '?')} + {len(skills)} skill(s)"
            else:
                rtarget = rcfg.get("script")
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
            init_counts = _count_statuses(init_ideas, self.results_dir, lake=self.lake)
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
            logger.error("Config hash cache rebuild failed: %s", e)
            notify("config_hash_failure", {"error": str(e)}, self.cfg)

        # Initialize code change detector (removed in v4.0)

        # Compute sealed file manifest for metric integrity
        sealed_files = cfg.get("sealed_files", [])
        if sealed_files:
            from orze.engine.sealed import compute_sealed_hashes, write_sealed_manifest
            hashes = compute_sealed_hashes(sealed_files)
            if hashes:
                write_sealed_manifest(self.results_dir, hashes)

        while self.running:
            self.iteration += 1
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            logger.info("--- Iteration %d [%s] ---", self.iteration, ts)

            # Hot-reload config every 10 iterations (~5 min)
            if self.iteration % 10 == 0:
                self._hot_reload_config()

            # 0a. Early heartbeat — keeps nodes UI alive even when
            #     iterations are slow (large results_dir scans).
            try:
                busy = self.slot_mgr.gpu_ids_in_use() | set(self.active_evals.keys())
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

            # 2b. Periodic orphan cleanup (every 10 iterations ≈ 5 min)
            if self.iteration % 10 == 0:
                try:
                    self._kill_orphans()
                except Exception:
                    pass

            # 2c. Periodic metric harvest (every 20 iterations ≈ 5 min).
            # Training scripts that log per-epoch metrics to stdout but
            # never emit metrics.json otherwise leave the leaderboard
            # blind to mid-run progress. Harvester fills in metrics.json
            # from train_output.log so professor/leaderboard see reality.
            # When the bundled regex defaults miss (exotic log formats),
            # orze-pro's pattern_inference can learn patterns via LLM
            # and cache them keyed by train_script mtime — one call per
            # new script, free thereafter.
            if self.iteration % 20 == 0:
                try:
                    from orze.engine.metric_harvester import harvest_running_ideas
                    mh_cfg = cfg.get("metric_harvest") or {}
                    if mh_cfg.get("enabled", True):
                        primary = (cfg.get("report") or {}).get(
                            "primary_metric", "map")
                        extra = mh_cfg.get("patterns") or []
                        maximize = mh_cfg.get("maximize", True)
                        inferrer = None
                        if mh_cfg.get("llm_fallback", True):
                            try:
                                from orze_pro.agents.pattern_inference import (
                                    infer_metric_patterns,
                                )
                                inf_model = mh_cfg.get(
                                    "inference_model", "haiku")
                                inf_timeout = int(mh_cfg.get(
                                    "inference_timeout", 60))
                                inf_bin = mh_cfg.get("claude_bin", "") or ""

                                def inferrer(script, text, metric,
                                             _fn=infer_metric_patterns,
                                             _m=inf_model,
                                             _t=inf_timeout,
                                             _b=inf_bin):
                                    return _fn(script, text, metric,
                                               model=_m, timeout=_t,
                                               claude_bin=_b)
                            except ImportError:
                                inferrer = None
                        ts_str = cfg.get("train_script")
                        ts_path = Path(ts_str) if ts_str else None
                        n = harvest_running_ideas(
                            self.results_dir, primary, extra,
                            maximize=maximize,
                            pattern_inferrer=inferrer,
                            train_script=ts_path)
                        if n > 0:
                            logger.info(
                                "Metric harvest: updated %d running idea(s)", n)
                except Exception as e:
                    logger.debug("Metric harvest failed: %s", e)

            # 3. Check active training processes (with health monitoring)
            finished = []
            if self.active:
                finished = check_active(self.active, self.results_dir,
                                        cfg, self.failure_counts,
                                        self.fix_counts)

            # 3-auto. After a SUCCESSFUL job finishes, check if GPU mode
            # should upgrade from exclusive to VRAM packing.
            # Critical: only check on success — failed jobs (exit code 2,
            # argparse errors) use 0 VRAM and would falsely trigger packing
            # mode, cascading into 90 launches per GPU.
            if finished and self._auto_gpu_mode:
                from orze.engine.gpu_slots import _query_all_gpu_usage
                # Find the first successful completion (has metrics.json
                # with status=COMPLETED and ran for >30 seconds)
                success_gpu = None
                for idea_id, gpu_key in finished:
                    metrics_path = self.results_dir / idea_id / "metrics.json"
                    if metrics_path.exists():
                        try:
                            m = json.loads(metrics_path.read_text(encoding="utf-8"))
                            if m.get("status") == "COMPLETED" and m.get("training_time", 0) > 30:
                                success_gpu = gpu_key
                                break
                        except (json.JSONDecodeError, OSError):
                            pass

                if success_gpu is not None:
                    try:
                        usage = _query_all_gpu_usage()
                        if usage:
                            gpu_id = int(str(success_gpu).split(":")[0]) if ":" in str(success_gpu) else success_gpu
                            if gpu_id in usage:
                                used, total = usage[gpu_id]
                                pct = used / total * 100 if total > 0 else 100
                                if pct < 30:
                                    logger.info(
                                        "Auto GPU mode: successful job used %d/%d MiB (%.0f%%) — "
                                        "upgrading to VRAM packing for higher throughput",
                                        used, total, pct)
                                    self.slot_mgr.mode = "vram"
                                    self.slot_mgr.max_jobs_per_gpu = max(
                                        int(90 / max(pct, 1)), 2)
                                    logger.info("  max_jobs_per_gpu set to %d",
                                                self.slot_mgr.max_jobs_per_gpu)
                                else:
                                    logger.info(
                                        "Auto GPU mode: successful job used %.0f%% VRAM — "
                                        "keeping exclusive mode", pct)
                        self._auto_gpu_mode = False  # only check once
                    except Exception:
                        self._auto_gpu_mode = False
                # Don't disable auto_gpu_mode on failures — wait for a
                # real success to make the decision.

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

            # 3c. Upgrade notification (user-triggered only, no auto-restart)
            # Sentinel-triggered upgrades (another node already pip-installed)
            # are handled in _check_upgrade_sentinel() above.

            # 4. Run agent roles (research, documenter, etc.)
            try:
                self._run_all_roles()
            except Exception as e:
                logger.error("Error in _run_all_roles: %s — continuing", e)
                notify("role_management_error", {"error": str(e)}, cfg)

            if not self.running:
                break

            # 4b. Mid-iteration heartbeat — keeps nodes alive during long iterations
            try:
                busy = self.slot_mgr.gpu_ids_in_use() | set(self.active_evals.keys())
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
            counts = _count_statuses(ideas, self.results_dir, lake=self.lake)
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
                    once_counts = _count_statuses(ideas, self.results_dir, lake=self.lake)
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
                        busy = self.slot_mgr.gpu_ids_in_use() | set(self.active_evals.keys())
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
        try:
            if self._leader_handle is not None:
                self._leader_handle.release()
        except Exception:
            pass
        logger.info("Exited after %d iterations.", self.iteration)
