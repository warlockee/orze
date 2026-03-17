"""Main loop phase methods for the Orze orchestrator.

CALLING SPEC:
    class OrzePhaseMixin:
        _sync_ideas(cfg) -> (ideas, unclaimed, skipped, raw_ideas)
        _launch_evals(finished, eval_finished, ideas) -> (eval_finished, backlog)
        _launch_training(unclaimed, disk_ok, ideas) -> free
        _report_and_notify(completed_rows, ideas, counts, eval_finished,
                           free, unclaimed, skipped, disk_ok, backlog) -> None

    Mixed into the Orze class to keep orchestrator.py under 800 LOC.
    Each method accesses orchestrator state via self.
"""

import json
import logging
import re
import shutil
import socket
import time
from pathlib import Path
from typing import Dict

import yaml

from orze.core.fs import _fs_lock, _fs_unlock, atomic_write
from orze.core.ideas import parse_ideas, expand_sweeps
from orze.engine.config_dedup import hash_config, load_hashes, save_hash
from orze.engine.evaluator import launch_eval, check_active_evals, run_eval, run_post_scripts
from orze.engine.failure import (
    _record_failure, get_skipped_ideas, _try_executor_fix, _reset_idea_for_retry,
)
from orze.engine.launcher import (
    launch, _get_checkpoint_dir, _write_failure,
)
from orze.engine.process import run_pre_script
from orze.engine.scheduler import claim, get_unclaimed, _count_statuses
from orze.hardware.gpu import get_gpu_memory_used, _eval_already_running
from orze.reporting.leaderboard import update_report, write_admin_cache
from orze.reporting.notifications import notify
from orze.reporting.state import (
    save_state, write_host_heartbeat, write_status_json,
)

logger = logging.getLogger("orze")


class OrzePhaseMixin:
    """Phase methods for the main orchestration loop."""

    def _sync_ideas(self, cfg):
        """Phase: sync ideas from ideas.md to lake, expand sweeps, build unclaimed queue."""
        raw_ideas = parse_ideas(cfg["ideas_file"])
        if self.lake:
            # Sync new ideas to DB queue
            db_ids = self.lake.get_all_ids()
            ingested_ids = []
            config_hashes = self._load_config_hashes()
            for idea_id, idea in raw_ideas.items():
                if idea_id not in db_ids:
                    # Config dedup: skip if overrides match a completed idea
                    override_hash = self._config_override_hash(
                        idea.get("config", {}))
                    existing_id = config_hashes.get(override_hash)
                    if existing_id:
                        logger.info(
                            "Skipping %s: config duplicate of %s",
                            idea_id, existing_id)
                        continue
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
        sweep_max = cfg.get("sweep", {}).get("max_combos", 20)
        queue_ideas = {}

        if self.lake:
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

        # On-demand reconcile: if queue returned ideas but none are
        # unclaimed, stale DB rows are blocking — reconcile and retry.
        if not unclaimed and queue_ideas and self.lake:
            n = self.lake.reconcile_statuses(str(self.results_dir))
            if n:
                logger.info("On-demand reconcile: cleared %d stale ideas", n)
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
                ideas = expand_sweeps(queue_ideas, max_combos=sweep_max)
                unclaimed = get_unclaimed(ideas, self.results_dir, skipped)

        if unclaimed:
            logger.info("Unclaimed queue (top 5): %s", unclaimed[:5])

        return ideas, unclaimed, skipped, raw_ideas

    def _launch_evals(self, finished, eval_finished, ideas):
        """Phase: launch evals for finished training, pending evals, and backlog."""
        cfg = self.cfg
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

        return eval_finished, backlog

    def _launch_training(self, unclaimed, disk_ok, ideas):
        """Phase: launch training on free GPUs, enforce sweep limits, circuit breaker."""
        cfg = self.cfg
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
                    if not claim(idea_id, self.results_dir, gpu,
                                 lake=self.lake):
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

        return free

    def _report_and_notify(self, completed_rows, ideas, counts,
                           eval_finished, free, unclaimed, skipped,
                           disk_ok, backlog):
        """Phase: heartbeat, notifications, status.json, admin cache, save state."""
        cfg = self.cfg

        # 9a. Retrospection hook (use completed_rows from report, not counts)
        try:
            self._run_retrospection(len(completed_rows) if completed_rows else 0)
        except Exception as e:
            logger.warning("Retrospection hook error: %s", e)

        # 9b. Notifications (fires for eval-finished ideas, metrics available)
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
        save_state(self.results_dir, self._build_state_dict())

