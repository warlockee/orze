"""Notification and reporting logic for finished experiments.

Calling spec
------------
    from orze.engine.reporter import NotificationProcessor

    proc = NotificationProcessor(results_dir, cfg, lake=lake)
    proc.load_state(state_dict)                       # restore from state.json
    saved = proc.get_state()                          # persist to state.json
    proc.process(finished, completed_rows, ideas,     # fire notifications
                 counts, active_count,
                 save_config_hash_fn,
                 build_machine_status_fn)

The processor owns plateau-detection state and periodic-report timing.
All notification delivery is delegated to ``orze.reporting.notifications.notify``.
"""

import json
import logging
import re
import time
from pathlib import Path

import yaml

from orze.core.fs import deep_get
from orze.reporting.leaderboard import _resolve_primary_metric
from orze.reporting.notifications import notify

logger = logging.getLogger("orze")


class NotificationProcessor:
    """Fires notifications for finished experiments.

    Owns plateau-detection counters and periodic-report timing so that
    the orchestrator can persist / restore them across restarts.
    """

    def __init__(self, results_dir: Path, cfg: dict, lake=None):
        self.results_dir = results_dir
        self.cfg = cfg
        self.lake = lake
        self._best_idea_id = None  # Optional[str]
        self._completions_since_best: int = 0
        self._plateau_notified: bool = False
        self._last_report_notify: float = 0.0

    def load_state(self, state: dict):
        """Restore persisted state from state.json."""
        self._best_idea_id = state.get("best_idea_id")
        self._completions_since_best = state.get("completions_since_best", 0)
        self._plateau_notified = state.get("plateau_notified", False)

    def get_state(self) -> dict:
        """Return state dict for persistence."""
        return {
            "best_idea_id": self._best_idea_id,
            "completions_since_best": self._completions_since_best,
            "plateau_notified": self._plateau_notified,
        }

    def process(self, finished: list, completed_rows: list, ideas: dict,
                counts: dict, active_count: int,
                save_config_hash_fn, build_machine_status_fn):
        """Fire notifications for finished experiments. Never raises."""
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

            # Build rank lookup and top-10 leaderboard
            rank_lookup, leaderboard = {}, []
            for rank, r in enumerate(completed_rows, 1):
                rank_lookup[r["id"]] = rank
                if rank <= 10:
                    leaderboard.append({"id": r["id"],
                                        "title": r.get("title", r["id"]),
                                        "value": r.get("primary_val")})

            view_lbs = self._build_view_leaderboards(cfg)
            row_lookup = {r["id"]: r for r in completed_rows}

            for idea_id, gpu in finished:
                self._notify_finished(
                    idea_id, gpu, cfg, primary, row_lookup, rank_lookup,
                    leaderboard, view_lbs, ideas, save_config_hash_fn)

            # New best detection + plateau tracking
            new_best = self._check_new_best(
                completed_rows, primary, leaderboard, view_lbs, cfg)
            n_completed = self._count_completed_in_batch(finished)
            if new_best:
                self._completions_since_best = 0
                self._plateau_notified = False
            else:
                self._completions_since_best += n_completed

            self._check_plateau(completed_rows, cfg)
            self._periodic_report(ncfg, cfg, primary, counts, active_count,
                                  leaderboard, view_lbs, build_machine_status_fn)
        except Exception as e:
            logger.warning("Notification processing error: %s", e)

    # -- internal helpers ------------------------------------------------

    def _build_view_leaderboards(self, cfg: dict) -> dict:
        view_leaderboards = {}
        for view in (cfg.get("report", {}).get("views") or []):
            vname = view.get("name")
            if not vname:
                continue
            vpath = self.results_dir / f"_leaderboard_{vname}.json"
            if not vpath.exists():
                continue
            try:
                vdata = json.loads(vpath.read_text(encoding="utf-8"))
                vtop = [{"id": e.get("idea_id", "?"), "title": e.get("title", ""),
                         "value": e.get("metric_value")}
                        for e in (vdata.get("top") or [])[:10]]
                if vtop:
                    view_leaderboards[vname] = {
                        "title": vdata.get("title", vname), "entries": vtop}
            except (json.JSONDecodeError, OSError):
                pass
        return view_leaderboards

    def _notify_finished(self, idea_id, gpu, cfg, primary, row_lookup,
                         rank_lookup, leaderboard, view_lbs, ideas,
                         save_config_hash_fn):
        m_path = self.results_dir / idea_id / "metrics.json"
        if not m_path.exists():
            return
        try:
            m = json.loads(m_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return

        status = m.get("status", "UNKNOWN")
        title = ideas.get(idea_id, {}).get("title", idea_id)

        if status == "COMPLETED":
            self._notify_completed(idea_id, title, m, cfg, primary,
                                   row_lookup, rank_lookup,
                                   leaderboard, view_lbs)
        elif status == "FAILED":
            notify("failed", {"idea_id": idea_id, "title": title,
                              "error": m.get("error", "unknown"),
                              "leaderboard": leaderboard,
                              "view_leaderboards": view_lbs}, cfg)

        if self.lake and status in ("COMPLETED", "FAILED"):
            self._archive_to_lake(idea_id, status, ideas, cfg)

        if status == "COMPLETED":
            try:
                rp = self.results_dir / idea_id / "resolved_config.yaml"
                if rp.exists():
                    rcfg = yaml.safe_load(rp.read_text(encoding="utf-8")) or {}
                    save_config_hash_fn(idea_id, rcfg)
            except Exception as exc:
                logger.debug("Config hash save failed for %s: %s",
                             idea_id, exc)

    def _notify_completed(self, idea_id, title, m, cfg, primary,
                          row_lookup, rank_lookup, leaderboard, view_lbs):
        row = row_lookup.get(idea_id, {})
        metric_val = row.get("primary_val") or m.get(primary)
        if metric_val is None:
            eval_file = cfg.get("eval_output", "eval_report.json")
            eval_path = self.results_dir / idea_id / eval_file
            if eval_path.exists():
                try:
                    ed = json.loads(eval_path.read_text(encoding="utf-8"))
                    metric_val = _resolve_primary_metric(cfg, eval_file, ed)
                except (json.JSONDecodeError, OSError,
                        KeyError, UnicodeDecodeError):
                    pass

        if metric_val is None:
            logger.warning(
                "Notification for %s has metric_val=None "
                "(row_pv=%s, m.get(%s)=%s, eval_exists=%s)",
                idea_id, row_lookup.get(idea_id, {}).get("primary_val"),
                primary, m.get(primary),
                (self.results_dir / idea_id /
                 cfg.get("eval_output", "eval_report.json")).exists())

        t_time = m.get("training_time") or None
        fmt_val = (f"{metric_val:.4f}"
                   if isinstance(metric_val, (int, float)) else metric_val)
        rank = rank_lookup.get(idea_id, None)

        # notify_top_n: only send "completed" notifications for top-N results.
        # Default 0 = notify all (backward compat). Set in orze.yaml:
        #   notifications:
        #     notify_top_n: 20
        top_n = (cfg.get("notifications") or {}).get("notify_top_n", 0)
        summary_only = (top_n > 0 and isinstance(rank, int) and rank > top_n)

        notify("completed", {
            "idea_id": idea_id, "title": title,
            "metric_name": primary, "metric_value": fmt_val,
            "training_time": t_time,
            "rank": rank if rank is not None else "?",
            "leaderboard": leaderboard,
            "view_leaderboards": view_lbs,
            "summary_only": summary_only,
        }, cfg)

    def _archive_to_lake(self, idea_id, status, ideas, cfg):
        try:
            idea_data = ideas.get(idea_id, {})
            config_yaml = ""
            raw_md = idea_data.get("raw", "")
            if idea_data.get("config"):
                config_yaml = yaml.dump(idea_data["config"],
                                        default_flow_style=False)
            eval_metrics = {}
            eval_file = cfg.get("eval_output", "eval_report.json")
            eval_path = self.results_dir / idea_id / eval_file
            if eval_path.exists():
                try:
                    ed = json.loads(eval_path.read_text(encoding="utf-8"))
                    em = ed.get("metrics", {})
                    for col in cfg.get("report", {}).get("columns", []):
                        src, key = col.get("source", ""), col.get("key", "")
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

            def _raw_field(field):
                match = re.search(
                    rf"\*\*{re.escape(field)}\*\*:\s*(.+)", raw_md)
                return match.group(1).strip() if match else None

            self.lake.insert(
                idea_id, idea_data.get("title", idea_id),
                config_yaml, raw_md,
                eval_metrics=eval_metrics or None,
                status=status.lower(),
                priority=idea_data.get("priority", "medium"),
                category=_raw_field("Category"),
                parent=_raw_field("Parent"),
                hypothesis=_raw_field("Hypothesis"))
        except Exception as exc:
            logger.warning("Failed to archive %s to lake: %s", idea_id, exc)

    def _check_new_best(self, completed_rows, primary, leaderboard,
                        view_lbs, cfg) -> bool:
        if not completed_rows:
            return False
        current_best = completed_rows[0]["id"]
        fired = False
        if (self._best_idea_id is not None
                and current_best != self._best_idea_id):
            best_val = completed_rows[0].get("primary_val")
            fmt = (f"{best_val:.4f}"
                   if isinstance(best_val, (int, float)) else best_val)
            notify("new_best", {
                "idea_id": current_best,
                "title": completed_rows[0]["title"],
                "metric_name": primary, "metric_value": fmt,
                "prev_best_id": self._best_idea_id,
                "leaderboard": leaderboard,
                "view_leaderboards": view_lbs,
            }, cfg)
            fired = True
        self._best_idea_id = current_best
        return fired

    def _count_completed_in_batch(self, finished: list) -> int:
        n = 0
        for idea_id, _ in finished:
            mp = self.results_dir / idea_id / "metrics.json"
            if not mp.exists():
                continue
            try:
                if json.loads(mp.read_text(encoding="utf-8")
                              ).get("status") == "COMPLETED":
                    n += 1
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                continue
        return n

    def _check_plateau(self, completed_rows, cfg):
        threshold = cfg.get("plateau_threshold", 50)
        if (threshold > 0
                and self._completions_since_best >= threshold
                and not self._plateau_notified):
            best_score = (completed_rows[0].get("primary_val")
                          if completed_rows else None)
            notify("plateau", {
                "message": (f"No improvement in {self._completions_since_best}"
                            f" ideas. Best: {best_score}"
                            f" ({self._best_idea_id})"),
                "best_id": self._best_idea_id,
                "since_best": self._completions_since_best,
                "threshold": threshold,
            }, cfg)
            self._plateau_notified = True

    def _periodic_report(self, ncfg, cfg, primary, counts, active_count,
                         leaderboard, view_lbs, build_machine_status_fn):
        interval = ncfg.get("report_interval", 0)
        if interval <= 0:
            return
        if time.time() - self._last_report_notify < interval:
            return
        notify("report", {
            "title": cfg["report"].get("title", "Report"),
            "completed": counts.get("COMPLETED", 0),
            "failed": counts.get("FAILED", 0),
            "active_count": active_count,
            "queued": counts.get("QUEUED", 0),
            "metric_name": primary,
            "leaderboard": leaderboard,
            "view_leaderboards": view_lbs,
            "machines": build_machine_status_fn(),
        }, cfg)
        self._last_report_notify = time.time()
