import logging
import os
import subprocess
from pathlib import Path

from orze.core.fs import tail_file
from orze.core.ideas import parse_ideas

logger = logging.getLogger("orze")


def _record_failure(failure_counts: dict, idea_id: str):
    """Increment failure count for an idea."""
    failure_counts[idea_id] = failure_counts.get(idea_id, 0) + 1


def get_skipped_ideas(failure_counts: dict, max_failures: int) -> set:
    """Return set of idea IDs that have exceeded max failure count."""
    if max_failures <= 0:
        return set()
    return {iid for iid, count in failure_counts.items()
            if count >= max_failures}


def _reset_idea_for_retry(idea_dir: Path):
    """Clean up an idea's result dir so it can be re-launched.

    Removes metrics.json and renames the old log for reference,
    but preserves claim.json (idea stays claimed by us).
    """
    metrics = idea_dir / "metrics.json"
    if metrics.exists():
        metrics.unlink(missing_ok=True)

    log = idea_dir / "train_output.log"
    if log.exists():
        attempt = 1
        while (idea_dir / f"train_output.attempt{attempt}.log").exists():
            attempt += 1
        log.rename(idea_dir / f"train_output.attempt{attempt}.log")


def _try_executor_fix(idea_id: str, error_text: str, results_dir: Path,
                      cfg: dict, fix_counts: dict) -> bool:
    """Spawn an LLM to diagnose and fix a failed idea.

    The LLM can modify the project's scripts, configs, or any other project
    files needed to make the idea succeed. It does NOT touch orze framework
    code.

    Returns True if the LLM reports a fix was applied (idea should be retried).
    """
    max_fix = cfg.get("max_fix_attempts", 0)
    if max_fix <= 0:
        return False

    attempts = fix_counts.get(idea_id, 0)
    if attempts >= max_fix:
        logger.info("[FIX] %s — exhausted %d fix attempts, giving up",
                     idea_id, attempts)
        return False

    attempt_num = attempts + 1

    idea_dir = results_dir / idea_id
    log_tail = tail_file(idea_dir / "train_output.log", 16384)

    # Read the idea config from ideas.md
    ideas_file = cfg.get("ideas_file", "ideas.md")
    idea_raw = ""
    try:
        ideas = parse_ideas(ideas_file)
        if idea_id in ideas:
            idea_raw = ideas[idea_id].get("raw", "")
    except Exception:
        pass

    train_script = cfg.get("train_script", "train.py")

    # Collect previous fix attempt logs so the LLM doesn't repeat itself
    prev_attempts_text = ""
    if attempt_num > 1:
        fix_log_dir = results_dir / "_fix_logs"
        parts = []
        for prev in range(1, attempt_num):
            prev_log = fix_log_dir / f"{idea_id}_attempt{prev}.log"
            if prev_log.exists():
                content = prev_log.read_text(encoding="utf-8")
                parts.append(f"### Attempt {prev}\n```\n{content[-2000:]}\n```")
        if parts:
            prev_attempts_text = (
                "\n\n## Previous Fix Attempts (ALREADY TRIED — do NOT repeat)\n"
                + "\n".join(parts)
            )

    prompt = f"""You are the executor for orze, an automated experiment orchestrator.
An experiment just failed. Your job: diagnose the error and fix whatever is needed
so the experiment can succeed on retry.

## Failed Experiment
- **Idea ID**: {idea_id}
- **Script**: {train_script}
- **Fix attempt**: {attempt_num} of {max_fix}

## Idea Config
```
{idea_raw[:3000]}
```

## Error
```
{error_text[:2000]}
```

## Full Log Tail (last 16KB)
```
{log_tail[-8000:]}
```
{prev_attempts_text}

## Your Task
1. Read the experiment script and any relevant project files to understand the error.
2. Diagnose the root cause.
3. Apply a minimal fix. Common patterns:
   - Wrong paths, missing files, empty inputs \u2192 fix paths or input handling
   - Resource exhaustion (memory, disk, etc.) \u2192 reduce resource usage in config
   - Dimension or type mismatches \u2192 fix the code to handle the config correctly
   - Missing dependencies \u2192 install or work around
   - Config handling errors \u2192 fix the script's config parsing/validation
4. After fixing, verify syntax: `python -c "import ast; ast.parse(open('FILE').read())"`
5. Report what you did.

## CRITICAL RULES
- You may modify ANY project file (scripts, configs, utilities, etc.)
- You may NOT modify files under `orze/` \u2014 that is the framework, not your scope
- You may NOT modify `ideas.md` \u2014 the idea config is fixed, adapt the code to handle it
- **CONCURRENT SAFETY**: Other experiments may be running right now using the same scripts.
  Your fix must not break them. Prefer fixes that are conditional on the specific config
  values (e.g., `if cfg.get("x") ...`) rather than changing default behavior.
  If you must change shared code, ensure the change is backward-compatible.
- Keep fixes minimal and targeted \u2014 don't refactor, just fix the specific error
- If the error is unfixable (e.g., corrupt data, impossible config), say "UNFIXABLE: <reason>"
  and exit without changes
"""

    logger.info("[FIX] %s — spawning LLM fix (attempt %d/%d): %s",
                 idea_id, attempt_num, max_fix, error_text[:100])

    fix_log_dir = results_dir / "_fix_logs"
    fix_log_dir.mkdir(parents=True, exist_ok=True)
    fix_log = fix_log_dir / f"{idea_id}_attempt{attempt_num}.log"

    try:
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        env.pop("CLAUDE_CODE_ENTRYPOINT", None)

        fix_cfg = cfg.get("executor_fix", {})
        claude_bin = fix_cfg.get("claude_bin") or "claude"
        model = fix_cfg.get("model") or "sonnet"
        fix_timeout = fix_cfg.get("timeout", 300)

        cmd = [
            claude_bin, "-p", prompt,
            "--dangerously-skip-permissions",
            "--output-format", "text",
            "--model", model,
        ]

        result = subprocess.run(
            cmd, capture_output=True, text=True,
            timeout=fix_timeout, env=env,
            cwd=str(results_dir.parent),
        )

        # LLM actually ran — count this attempt
        fix_counts[idea_id] = attempts + 1

        response = result.stdout[-5000:] if result.stdout else ""
        fix_log.write_text(
            f"=== Attempt {attempt_num} for {idea_id} ===\n"
            f"Error: {error_text[:500]}\n\n"
            f"=== LLM Response ===\n{response}\n",
            encoding="utf-8",
        )

        if "UNFIXABLE" in response.upper():
            logger.info("[FIX] %s — LLM says unfixable: %s",
                         idea_id, response[:200])
            return False

        logger.info("[FIX] %s — LLM applied fix (attempt %d), will retry",
                     idea_id, attempt_num)
        return True

    except subprocess.TimeoutExpired:
        # Timed out — still count it (LLM may have made partial changes)
        fix_counts[idea_id] = attempts + 1
        logger.warning("[FIX] %s — LLM fix timed out (attempt %d)",
                        idea_id, attempt_num)
        return False
    except FileNotFoundError:
        # Don't count — Claude CLI not installed, no attempt was made
        logger.warning("[FIX] Claude CLI not found — skipping executor fix")
        return False
    except Exception as e:
        logger.error("[FIX] %s — LLM fix error: %s", idea_id, e)
        return False
