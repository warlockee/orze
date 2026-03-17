"""Agent role lifecycle for Orze.

CALLING SPEC:
    build_claude_cmd(role_cfg, template_vars) -> list[str] | None
    build_research_cmd(role_cfg, template_vars, cfg) -> list[str]
    run_role_step(role_name, role_cfg, ctx) -> None
    run_all_roles(ctx) -> None
    run_role_once(role_name, ctx) -> None

    ctx is a RoleContext dataclass containing:
        cfg, results_dir, gpu_ids, active_roles, role_states,
        failure_counts, fix_counts, iteration
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

from orze.engine.process import RoleProcess, _new_process_group
from orze.engine.launcher import _format_args
from orze.engine.scheduler import get_unclaimed, _count_statuses
from orze.engine.failure import get_skipped_ideas
from orze.engine.roles import check_active_roles
from orze.core.fs import _fs_lock, _fs_unlock
from orze.core.ideas import parse_ideas
from orze.reporting.notifications import notify

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------

@dataclass
class RoleContext:
    cfg: dict
    results_dir: Path
    gpu_ids: list
    active_roles: dict    # mutated in-place
    role_states: dict     # mutated in-place
    failure_counts: dict  # read-only
    fix_counts: dict      # read-only
    iteration: int

# ---------------------------------------------------------------------------
# Command builders
# ---------------------------------------------------------------------------

def build_claude_cmd(
    role_cfg: dict,
    template_vars: dict,
) -> Optional[List[str]]:
    """Build a Claude CLI command for mode: claude."""
    # Skills path (composable) or legacy rules_file path
    if "skills" in role_cfg:
        from orze.skills.loader import compose_skills
        project_root = Path(template_vars.get("results_dir", ".")).parent
        prompt = compose_skills(role_cfg, project_root, template_vars=template_vars)
        if not prompt:
            logger.warning("No skills produced any content")
            return None
    else:
        rules_file = role_cfg.get("rules_file")
        if not rules_file:
            return None
        rules_path = Path(rules_file)
        if not rules_path.exists():
            logger.warning("Research rules file not found: %s", rules_file)
            return None
        rules_content = rules_path.read_text(encoding="utf-8")
        prompt = rules_content
        for k, v in template_vars.items():
            prompt = prompt.replace(f"{{{k}}}", str(v))

    claude_bin = role_cfg.get("claude_bin") or "claude"
    cmd = [claude_bin, "-p", prompt]

    # --model (e.g., sonnet, opus, haiku)
    model = role_cfg.get("model")
    if model:
        cmd.extend(["--model", model])

    # --allowedTools (default: local tools only)
    allowed_tools = role_cfg.get("allowed_tools") or "Read,Write,Edit,Glob,Grep,Bash"
    cmd.extend(["--allowedTools", str(allowed_tools)])

    # --output-format
    output_format = role_cfg.get("output_format") or "text"
    cmd.extend(["--output-format", str(output_format)])

    # Any extra CLI args
    cmd.extend(_format_args(role_cfg.get("claude_args") or [],
                            template_vars))

    return cmd

def build_research_cmd(
    role_cfg: dict,
    template_vars: dict,
    cfg: dict,
) -> List[str]:
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
    python = cfg.get("python", sys.executable)
    # research_agent.py lives in the agents directory
    agent_script = Path(__file__).parent.parent / "agents" / "research.py"

    cmd = [python, str(agent_script)]
    cmd.extend(["-c", str(cfg.get("_config_path", "orze.yaml"))])
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
    if "skills" in role_cfg:
        from orze.skills.loader import compose_skills
        project_root = Path(template_vars.get("results_dir", ".")).parent
        composed = compose_skills(role_cfg, project_root, template_vars=None)
        if composed:
            results_dir = Path(template_vars["results_dir"])
            tmp = results_dir / f"_skill_composed_{template_vars.get('role_name', 'research')}.md"
            tmp.write_text(composed, encoding="utf-8")
            cmd.extend(["--rules-file", str(tmp)])
    elif role_cfg.get("rules_file"):
        cmd.extend(["--rules-file", str(role_cfg["rules_file"])])

    # Pass lake DB path so research agent can query historical patterns
    lake_path = Path(template_vars["ideas_file"]).parent / "idea_lake.db"
    if lake_path.exists():
        cmd.extend(["--lake-db", str(lake_path)])

    # Pass retrospection file if it exists
    results_dir = Path(template_vars["results_dir"])
    retro_file = results_dir / "_retrospection.txt"
    if retro_file.exists():
        cmd.extend(["--retrospection-file", str(retro_file)])

    return cmd

# ---------------------------------------------------------------------------
# Step / launch
# ---------------------------------------------------------------------------

def run_role_step(role_name: str, role_cfg: dict, ctx: RoleContext) -> None:
    """Launch agent role if not running and cooldown elapsed (non-blocking).

    Supports three modes:
      - mode: script  -- run a Python script
      - mode: claude  -- run Claude CLI with a rules/prompt file
      - mode: research -- run built-in LLM research agent
    """
    # Skip if already running
    if role_name in ctx.active_roles:
        return

    mode = role_cfg.get("mode", "script")
    if mode == "script" and not role_cfg.get("script"):
        return
    if mode == "claude" and not role_cfg.get("rules_file") and not role_cfg.get("skills"):
        return
    if mode == "research" and not role_cfg.get("backend"):
        return

    # Per-role cooldown (with adaptive producer-consumer matching)
    role_state = ctx.role_states.setdefault(
        role_name, {"cycles": 0, "last_run_time": 0.0})
    cooldown = role_cfg.get("cooldown", 300)
    elapsed = time.time() - role_state["last_run_time"]

    # Adaptive cooldown: if queue is nearly empty, skip cooldown to
    # keep GPUs fed. Only applies to the research role.
    queue_starving = False
    if role_name == "research" and elapsed >= 60:
        try:
            ideas = parse_ideas(ctx.cfg["ideas_file"])
            skipped = get_skipped_ideas(
                ctx.failure_counts,
                ctx.cfg.get("max_idea_failures", 0))
            n_unclaimed = len(get_unclaimed(
                ideas, ctx.results_dir, skipped))
            n_gpus = len(ctx.gpu_ids)

            # --- Hard queue cap: skip research entirely ---
            max_queue = ctx.cfg.get("max_queue_size", 500)
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
            patience = ctx.cfg.get("convergence_patience", 0)
            if patience > 0:
                best_val = role_state.get("_best_metric_val")
                best_at = role_state.get("_best_metric_at", 0)
                counts = _count_statuses(ideas, ctx.results_dir)
                n_completed = counts.get("COMPLETED", 0)

                # Read current best from completed rows
                primary = ctx.cfg["report"].get(
                    "primary_metric", "test_accuracy")
                sort_desc = ctx.cfg["report"].get(
                    "sort", "descending") == "descending"
                cur_best = None
                for d in ctx.results_dir.iterdir():
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
    lock_dir = ctx.results_dir / f"_{role_name}_lock"
    if not _fs_lock(lock_dir, stale_seconds=timeout + 60):
        logger.debug("%s lock held by another host, skipping", role_name)
        return

    # Template variables (shared across all roles)
    ideas = parse_ideas(ctx.cfg["ideas_file"])
    counts = _count_statuses(ideas, ctx.results_dir)
    template_vars = {
        "ideas_file": ctx.cfg["ideas_file"],
        "results_dir": str(ctx.results_dir),
        "cycle": role_state["cycles"] + 1,
        "gpu_count": len(ctx.gpu_ids),
        "completed": counts.get("COMPLETED", 0),
        "queued": counts.get("QUEUED", 0),
        "role_name": role_name,
    }

    # Build command based on mode
    if mode == "claude":
        cmd = build_claude_cmd(role_cfg, template_vars)
        if not cmd:
            _fs_unlock(lock_dir)
            return
    elif mode == "research":
        cmd = build_research_cmd(role_cfg, template_vars, ctx.cfg)
    else:
        python = ctx.cfg.get("python", sys.executable)
        cmd = [python, role_cfg["script"]]
        cmd.extend(_format_args(role_cfg.get("args") or [], template_vars))

    # Environment
    env = os.environ.copy()
    env.pop("CLAUDECODE", None)  # Allow nested Claude CLI sessions
    env.pop("CLAUDE_CODE_ENTRYPOINT", None)
    for k, v in (ctx.cfg.get("train_extra_env") or {}).items():
        env[k] = str(v)
    for k, v in (role_cfg.get("env") or {}).items():
        env[k] = str(v)

    # Per-role log directory
    log_dir_name = role_cfg.get("log_dir") or f"_{role_name}_logs"
    log_dir = ctx.results_dir / log_dir_name
    log_dir.mkdir(parents=True, exist_ok=True)
    cycle_num = role_state["cycles"] + 1
    log_path = log_dir / f"cycle_{cycle_num:03d}.log"

    logger.info("Running %s [%s] (cycle %d)...",
                 role_name, mode, cycle_num)

    # Protect ideas.md: snapshot size before research role runs
    ideas_file = Path(ctx.cfg.get("ideas_file", "ideas.md"))
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
        ctx.active_roles[role_name] = RoleProcess(
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

# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def run_all_roles(ctx: RoleContext) -> None:
    """Check active roles and launch new ones (non-blocking)."""
    # Check active roles
    finished = check_active_roles(
        ctx.active_roles,
        ideas_file=ctx.cfg.get("ideas_file", "ideas.md"))

    # Collect per-role results for consolidated notification
    role_contributions: Dict[str, int] = {}  # label -> new_ideas count
    any_ideas_modified = False

    for role_name, success in finished:
        role_state = ctx.role_states.setdefault(
            role_name, {"cycles": 0, "last_run_time": 0.0})
        role_state["last_run_time"] = time.time()
        role_state["cycles"] = role_state.get("cycles", 0) + 1

        if success:
            role_state["consecutive_failures"] = 0
            # Output validation: warn if ideas file wasn't modified
            ideas_file = Path(ctx.cfg.get("ideas_file", "ideas.md"))
            ideas_modified = False
            if ideas_file.exists():
                ideas_age = time.time() - ideas_file.stat().st_mtime
                role_timeout = (ctx.cfg.get("roles") or {}).get(
                    role_name, {}).get("timeout", 600)
                ideas_modified = ideas_age <= role_timeout
                if not ideas_modified:
                    logger.warning("%s completed successfully but ideas file "
                                   "was not modified (last change %.0fs ago)",
                                   role_name, ideas_age)
            if ideas_modified:
                ideas_now = parse_ideas(ctx.cfg["ideas_file"])
                prev_count = role_state.get("_prev_idea_count", 0)
                new_ideas = max(0, len(ideas_now) - prev_count)
                role_state["_prev_idea_count"] = len(ideas_now)
                any_ideas_modified = True
                # Derive a short label from the role's model config
                role_cfg = (ctx.cfg.get("roles") or {}).get(role_name, {})
                model = role_cfg.get("model", role_name)
                # Shorten model name: "claude-opus-4-6" -> "opus", "gemini-2.5-pro" -> "gemini"
                label = model
                for prefix in ("claude-", "gemini-"):
                    if label.startswith(prefix):
                        label = label[len(prefix):]
                        break
                label = label.split("-")[0].split(".")[0]  # first part
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
        ideas_now = parse_ideas(ctx.cfg["ideas_file"])
        n_queued = len(get_unclaimed(ideas_now, ctx.results_dir, set()))
        total_new = sum(role_contributions.values())
        # Build breakdown string: "5 <opus> + 10 <gemini>"
        parts = [f"{n} <{lbl}>" for lbl, n in role_contributions.items() if n > 0]
        breakdown = " + ".join(parts) if parts else f"{total_new}"
        notify("role_summary", {
            "role": "researcher",
            "new_ideas": total_new,
            "breakdown": breakdown,
            "queued": n_queued,
        }, ctx.cfg)

    # Launch new roles if not running
    for role_name, role_cfg in (ctx.cfg.get("roles") or {}).items():
        if isinstance(role_cfg, dict):
            run_role_step(role_name, role_cfg, ctx)


def run_role_once(role_name: str, ctx: RoleContext) -> None:
    """Run a single agent role synchronously, then exit."""
    roles = ctx.cfg.get("roles") or {}
    if role_name not in roles:
        logger.error("Role '%s' not found in config. Available: %s",
                     role_name, list(roles.keys()))
        return
    role_cfg = roles[role_name]
    if not isinstance(role_cfg, dict):
        logger.error("Role '%s' config is not a dict", role_name)
        return

    logger.info("Running role '%s' once...", role_name)
    run_role_step(role_name, role_cfg, ctx)

    # Wait for it to finish
    while role_name in ctx.active_roles:
        time.sleep(2)
        finished = check_active_roles(
            ctx.active_roles,
            ideas_file=ctx.cfg.get("ideas_file", "ideas.md"))
        for rn, success in finished:
            if success:
                logger.info("Role '%s' completed successfully", rn)
            else:
                logger.warning("Role '%s' failed", rn)
