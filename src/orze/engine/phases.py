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
import sys
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
from orze.engine.sealed import load_sealed_manifest, verify_sealed_files


def _load_verified(results_dir: Path, report_cfg: dict):
    """Load verified full-scale results from file if configured."""
    vpath = report_cfg.get("verified_results")
    if not vpath:
        return None
    p = results_dir / vpath if not Path(vpath).is_absolute() else Path(vpath)
    # Also try relative to cwd
    if not p.exists():
        p = Path(vpath)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None

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
                    # Validation moved to launch time (catches ALL idea sources)
                    idea_cfg = idea.get("config", {})
                    self.lake.insert(
                        idea_id, idea["title"], yaml.dump(idea_cfg),
                        raw_text,
                        status="queued",
                        priority=raw_pri,
                        category=_raw_f("Category"),
                        parent=_raw_f("Parent"),
                        hypothesis=_raw_f("Hypothesis"),
                        approach_family=idea.get("approach_family", _raw_f("Approach Family") or "other"),
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
                        # Record the legitimate wipe so the corruption
                        # guard knows this shrink was designed, not rogue.
                        from orze.engine.roles import mark_ingest
                        mark_ingest(Path(cfg["ideas_file"]))
                        # Update pre-snapshots on active roles so the corruption
                        # guard doesn't false-positive on the legitimate wipe.
                        # Also credit research-writer roles with the ingested
                        # count so they don't trip the "ideas.md was not
                        # modified" soft-failure check — they appended, we
                        # just consumed it before they exited.
                        new_size = Path(cfg["ideas_file"]).stat().st_size
                        ingested_n = len(ingested_ids)
                        for rp in self.active_roles.values():
                            rp.ideas_pre_size = new_size
                            rp.ideas_pre_count = 0
                            if getattr(rp, "writes_ideas_file", True) and ingested_n:
                                rp.ideas_consumed_during_run += ingested_n
                    except Exception as e:
                        logger.warning("Failed to consume ideas.md: %s", e)
                    finally:
                        _fs_unlock(ideas_lock)
                else:
                    logger.debug("Skipping ideas.md consumption (lock held)")

        # 5a-pre. Inline Tier 1 filter (orze-pro): skip garbage before it's claimed
        try:
            from orze.extensions import get_extension
            idea_filter = get_extension("idea_filter")
            if idea_filter:
                idea_filter.filter_queued_ideas(self.results_dir)
        except Exception:
            pass  # pro not installed or filter failed — continue normally

        # 5a-pre2. Tier 2 SOP: auto-generate ideas from portfolios (orze-pro)
        # Validates each backbone's train_script BEFORE generating ideas.
        # If validation fails, triggers thinker/professor to implement.
        try:
            from orze.extensions import get_extension
            _sop_t2 = get_extension("sop_tier2")
            if _sop_t2 and self.lake:
                load_portfolios = _sop_t2.load_portfolios
                load_method_specs = _sop_t2.load_method_specs
                generate_portfolio_ideas = _sop_t2.generate_portfolio_ideas

                portfolios = load_portfolios(self.results_dir)
                methods = load_method_specs(self.results_dir)

                for portfolio in portfolios:
                    new_ideas = generate_portfolio_ideas(
                        portfolio, self.results_dir, methods)
                    for idea in new_ideas:
                        self.lake.insert(
                            idea["idea_id"], idea["title"],
                            yaml.dump(idea["config"]),
                            "", status="queued",
                            priority=idea.get("priority", "high"),
                            approach_family=idea.get("approach_family", "portfolio"),
                        )
                    if new_ideas:
                        # Track generated IDs in portfolio file
                        pf = portfolio.get("_file")
                        if pf:
                            portfolio.setdefault("generated_ideas", [])
                            portfolio["generated_ideas"].extend(
                                [i["idea_id"] for i in new_ideas])
                            try:
                                Path(pf).write_text(
                                    yaml.dump({k: v for k, v in portfolio.items()
                                               if k != "_file"},
                                              default_flow_style=False),
                                    encoding="utf-8")
                            except OSError:
                                pass
                        logger.info("Portfolio '%s': generated %d ideas",
                                    portfolio.get("name", "?"), len(new_ideas))
                    # Launch-time validate_idea catches bad configs — no need
                    # for pre-generation validation here.
        except ImportError:
            pass
        except Exception as e:
            logger.debug("Portfolio generation skipped: %s", e)

        # 5a. Expand sweeps and find unclaimed work
        sweep_max = cfg.get("sweep", {}).get("max_combos", 20)
        queue_ideas = {}

        if self.lake:
            for r in self.lake.get_queue(limit=100):
                try:
                    cfg_parsed = yaml.safe_load(r["config"]) or {}
                except yaml.YAMLError:
                    cfg_parsed = {}
                # Pre-filter: skip ideas with missing strategy files BEFORE
                # sweep expansion. This prevents expanding a broken idea into
                # N sub-runs that all get skipped individually every iteration.
                strategy_name = cfg_parsed.get("strategy")
                if strategy_name:
                    strategy_path = Path("strategies") / f"{strategy_name}.py"
                    if not strategy_path.exists():
                        try:
                            self.lake.set_status(r["idea_id"], "skipped")
                            logger.warning("Pre-filter: skipped %s (missing %s)",
                                           r["idea_id"], strategy_path)
                        except Exception:
                            pass
                        continue
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
        unclaimed = get_unclaimed(ideas, self.results_dir, skipped, lake=self.lake)

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
                unclaimed = get_unclaimed(ideas, self.results_dir, skipped, lake=self.lake)

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
                            if hasattr(self, 'slot_mgr'):
                                eval_busy = (self.slot_mgr.gpu_ids_in_use()
                                             | set(self.active_evals.keys()))
                            else:
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
            if hasattr(self, 'slot_mgr'):
                eval_busy = (self.slot_mgr.gpu_ids_in_use()
                             | set(self.active_evals.keys()))
            else:
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
                if hasattr(self, 'slot_mgr'):
                    eval_busy = (self.slot_mgr.gpu_ids_in_use()
                                 | set(self.active_evals.keys()))
                else:
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

        # Verify sealed file integrity before launching any training
        sealed_files = cfg.get("sealed_files", [])
        if sealed_files:
            manifest = load_sealed_manifest(self.results_dir)
            if manifest:
                changed = verify_sealed_files(sealed_files, manifest)
                if changed:
                    logger.error(
                        "Sealed file integrity check FAILED: %d file(s) changed: %s",
                        len(changed), ", ".join(changed))
                    notify("sealed_file_changed", {
                        "changed_files": changed,
                        "message": (f"{len(changed)} sealed file(s) changed since startup. "
                                    "Training will proceed but results may be inconsistent."),
                    }, cfg)

        eval_gpus = set(self.active_evals.keys())
        if hasattr(self, 'slot_mgr'):
            free = self.slot_mgr.free_gpu_ids(exclude=eval_gpus)
        else:
            busy_gpus = set(self.active.keys()) | eval_gpus
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

        # Emergency GC: if disk is low and GC is configured, try to free space now
        if not disk_ok:
            gc_cfg = cfg.get("gc", {})
            if gc_cfg.get("enabled"):
                try:
                    from orze.agents.orze_gc import run_gc
                    report_cfg = cfg.get("report") or {}
                    lake_path = Path(cfg.get("idea_lake_db") or Path(cfg.get("results_dir", "orze_results")) / "idea_lake.db")
                    # Collect idea IDs from all currently running training
                    # and eval processes so GC never deletes their checkpoints.
                    running_ids = set()
                    for tp in self.active.values():
                        running_ids.add(tp.idea_id)
                    for ep in self.active_evals.values():
                        running_ids.add(ep.idea_id)
                    logger.warning("Emergency GC: disk low, running checkpoint cleanup "
                                   "(protecting %d running experiments)", len(running_ids))
                    run_gc(
                        results_dir=self.results_dir,
                        checkpoints_dir=Path(gc_cfg["checkpoints_dir"]) if gc_cfg.get("checkpoints_dir") else None,
                        primary_metric=report_cfg.get("primary_metric", ""),
                        lake_db_path=lake_path if lake_path.exists() else None,
                        keep_top=gc_cfg.get("keep_top", 50),
                        keep_recent=gc_cfg.get("keep_recent", 20),
                        min_free_gb=0,  # force run, disk is already low
                        extra_keep_ids=running_ids,
                    )
                except Exception as e:
                    logger.warning("Emergency GC failed: %s", e)

        if unclaimed and free and disk_ok:
            # With multi-slot scheduling, keep launching until all slots are full
            max_launches = len(free) * getattr(getattr(self, 'slot_mgr', None), 'slots_per_gpu', 1)
            launch_count = 0
            while unclaimed and launch_count < max_launches:
                # Re-check free GPUs each iteration (slots fill up)
                if hasattr(self, 'slot_mgr'):
                    free = self.slot_mgr.free_gpu_ids(exclude=set(self.active_evals.keys()))
                if not free:
                    break
                gpu = free[0]  # least-loaded GPU (sorted by most free slots)
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
                    # Write idea config so train scripts can read it.
                    # The only sanitization needed here is the LLM-generated
                    # "collapsed" key pattern: {"epochs: 40": None}, which
                    # research agents occasionally emit when their response
                    # parser eats the ": " between key and value.
                    #
                    # Do NOT strip dict/list values — research agents often
                    # use structured config intentionally, e.g.
                    #   picks:
                    #     SmolTalk: 1.0
                    # which is a config-driven script's main knob. An earlier
                    # version of this code dropped those silently; every
                    # first-user idea that nested a dict under a key lost that
                    # key and had to be debugged at the train-script level.
                    idea_cfg = ideas.get(idea_id, {}).get("config", {})
                    flat_cfg = {}
                    if idea_cfg:
                        for k, v in idea_cfg.items():
                            if v is None and ": " in str(k):
                                parts = str(k).split(": ", 1)
                                try:
                                    flat_cfg[parts[0]] = yaml.safe_load(parts[1])
                                except Exception:
                                    flat_cfg[parts[0]] = parts[1]
                                logger.warning(
                                    "Fixed malformed config key %r -> %s: %s",
                                    k, parts[0], flat_cfg[parts[0]])
                            else:
                                flat_cfg[k] = v
                        atomic_write(
                            self.results_dir / idea_id / "idea_config.yaml",
                            yaml.dump(flat_cfg,
                                      default_flow_style=False))
                    # Write sweep config for sub-runs (legacy compat)
                    if ideas.get(idea_id, {}).get("_sweep_parent"):
                        atomic_write(
                            self.results_dir / idea_id / "sweep_config.yaml",
                            yaml.dump(idea_cfg,
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
                    # SOP: validate idea config right before launch (catches all sources)
                    # Use flat_cfg (malformed-key-cleaned) if available, else raw config.
                    # flat_cfg is built at lines 518-535 above, which fixes LLM-generated
                    # collapsed keys like "epochs: 40" → {"epochs": 40}.
                    try:
                        from orze.extensions import get_extension
                        _sops = get_extension("sops")
                        if _sops:
                            idea_cfg = flat_cfg if flat_cfg else ideas.get(idea_id, {}).get("config", {})
                            ts = idea_cfg.get("train_script", cfg.get("train_script", ""))
                            if ts and idea_cfg:
                                is_valid, err_msg = _sops.validate_idea(
                                    ts, idea_cfg, cfg.get("python", sys.executable))
                                if not is_valid:
                                    logger.warning("Skipping %s at launch: %s",
                                                   idea_id, err_msg)
                                    _write_failure(
                                        self.results_dir / idea_id,
                                        f"Config validation failed: {err_msg}")
                                    if self.lake:
                                        self.lake.set_status(idea_id, "skipped")
                                    # Trigger engineer once per (script, missing_key_set).
                                    # Previously keyed only by script: once fired for
                                    # train.py, no further unrecognized-arg rejection
                                    # could re-trigger it even with a brand-new missing
                                    # key, so research proposed 251 ideas across 3+
                                    # new config keys and only 2 engineer triggers
                                    # fired. Parse the missing keys out of err_msg
                                    # so each novel schema gap gets one attempt.
                                    if not hasattr(self, '_impl_triggered'):
                                        self._impl_triggered = set()
                                    import re as _re
                                    _keys = _re.findall(
                                        r'unrecognized args[^:]*:\s*(.*)$',
                                        err_msg)
                                    if _keys:
                                        _key_set = frozenset(
                                            k.strip() for k in _keys[0].split(",")
                                            if k.strip())
                                    else:
                                        _key_set = frozenset()
                                    _sig = (ts, _key_set)
                                    if _sig not in self._impl_triggered:
                                        self._impl_triggered.add(_sig)
                                        _sop_t2 = get_extension("sop_tier2")
                                        if _sop_t2 and hasattr(_sop_t2, 'trigger_implementation'):
                                            methods = _sop_t2.load_method_specs(self.results_dir)
                                            _sop_t2.trigger_implementation(
                                                {"train_script": ts, "name": ts},
                                                next(iter(methods.values()), {}),
                                                err_msg, self.results_dir)
                                    continue
                    except Exception:
                        pass

                    logger.info("Launching %s on GPU %s: %s",
                                idea_id, gpu,
                                ideas[idea_id]["title"][:50])
                    try:
                        tp = launch(idea_id, gpu, self.results_dir, cfg)
                    except Exception as e:
                        logger.error("Failed to launch %s on GPU %s: %s",
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
                    try:
                        self.active[gpu] = tp
                    except RuntimeError as slot_err:
                        # Race: capacity check passed at target selection but
                        # failed at registration (system-load throttle kicked in
                        # between launch() and assign()). Don't crash the
                        # orchestrator — terminate the orphan subprocess and
                        # retry the idea later.
                        logger.warning(
                            "[SLOT-RACE] %s on GPU %s: %s — terminating orphan "
                            "and deferring idea", idea_id, gpu, slot_err)
                        try:
                            from orze.engine.process import _terminate_and_reap
                            _terminate_and_reap(tp.process, f"orphan {idea_id}")
                            tp.close_log()
                        except Exception as cleanup_err:
                            logger.warning("Cleanup after slot-race failed: %s",
                                           cleanup_err)
                        if self.lake:
                            # Return idea to queue so a later iteration retries
                            self.lake.set_status(idea_id, "queued")
                        continue
                    base_id = idea_id
                    if "-ht-" in base_id:
                        base_id = base_id.split("-ht-", 1)[0]
                    elif "~" in base_id:
                        base_id = base_id.split("~", 1)[0]

                    if base_id != idea_id:
                        sweep_counts[base_id] = sweep_counts.get(base_id, 0) + 1
                    launched = True
                    launch_count += 1
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
            first_hb = (self._last_heartbeat == self._start_time)
            if first_hb or now_hb - self._last_heartbeat >= heartbeat_interval:
                uptime_s = int(now_hb - self._start_time)
                h, rem = divmod(uptime_s, 3600)
                m = rem // 60
                uptime_str = f"{h}h{m:02d}m" if h else f"{m}m"
                busy_gpus = self.slot_mgr.gpu_ids_in_use() | set(self.active_evals.keys())
                n_free = len([g for g in self.gpu_ids if g not in busy_gpus])
                # Best result for heartbeat
                hb_best_id = None
                hb_best_val = None
                hb_best_title = None
                if completed_rows:
                    primary = cfg["report"].get("primary_metric",
                                                "test_accuracy")
                    top = completed_rows[0]
                    hb_best_id = top.get("id", "?")
                    hb_best_val = top.get("primary_val")
                    hb_best_title = (top.get("title") or "")[:40]
                    hb_best_metric = cfg["report"].get(
                        "primary_metric", "score")

                n_running = len(self.active) + len(self.active_evals)

                # Collect per-dataset breakdown from best result
                hb_best_details = {}
                if completed_rows:
                    top = completed_rows[0]
                    for col in (cfg.get("report", {}).get("columns") or []):
                        k = col.get("key", "")
                        if k != primary and k in top and isinstance(top[k], (int, float)):
                            label = col.get("label", k)
                            hb_best_details[label] = top[k]

                # GPU info from nvidia-smi
                try:
                    from orze.hardware.gpu import _query_gpu_details
                    hb_gpu_info = _query_gpu_details()
                except Exception:
                    hb_gpu_info = []

                # Model name from base config
                hb_model = (cfg.get("base_config_data") or {}).get(
                    "model_path", cfg.get("model_name", ""))
                if hb_model and "/" in hb_model:
                    hb_model = hb_model.rstrip("/").rsplit("/", 1)[-1]

                # Active runs info for heartbeat
                hb_active_runs = []
                now_ts = time.time()
                for gpu_id, tp in sorted(self.active.items()):
                    elapsed = (now_ts - tp.start_time) / 60.0
                    hb_active_runs.append({
                        "idea_id": tp.idea_id,
                        "gpu": gpu_id,
                        "elapsed_min": round(elapsed, 1),
                    })
                for gpu_id, tp in sorted(self.active_evals.items()):
                    elapsed = (now_ts - tp.start_time) / 60.0
                    hb_active_runs.append({
                        "idea_id": tp.idea_id,
                        "gpu": gpu_id,
                        "elapsed_min": round(elapsed, 1),
                        "phase": "eval",
                    })

                notify("heartbeat", {
                    "host": socket.gethostname(),
                    "iteration": self.iteration,
                    "uptime": uptime_str,
                    "training": len(self.active),
                    "eval": len(self.active_evals),
                    "running": n_running,
                    "free": n_free,
                    "total_gpus": len(self.gpu_ids),
                    "completed": counts.get("COMPLETED", 0),
                    "queued": counts.get("QUEUED", 0),
                    "failed": counts.get("FAILED", 0),
                    "eval_backlog": len(backlog),
                    "rate": ("just started" if first_hb else
                             f"{counts.get('COMPLETED', 0) - self._hb_completed_count}"
                             f" since last heartbeat"),
                    "heartbeat_interval": heartbeat_interval,
                    "best_id": hb_best_id,
                    "best_val": hb_best_val,
                    "best_title": hb_best_title,
                    "best_metric": hb_best_metric if completed_rows else None,
                    "best_details": hb_best_details or None,
                    "gpu_info": hb_gpu_info or None,
                    "model_name": hb_model or None,
                    "target": cfg.get("report", {}).get("target"),
                    "sort": cfg.get("report", {}).get("sort", "descending"),
                    "estimate_label": cfg.get("report", {}).get("estimate_label"),
                    "estimate_warning": cfg.get("report", {}).get("estimate_warning"),
                    "timeout": cfg.get("timeout", 21600),
                    "verified": _load_verified(self.results_dir, cfg.get("report", {})),
                    "active_runs": hb_active_runs or None,
                    "next_queue": [
                        {"id": uid,
                         "title": (ideas.get(uid, {}).get("title", "") or "")[:40]}
                        for uid in unclaimed[:5]
                    ] or None,
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

