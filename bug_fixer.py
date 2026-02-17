#!/usr/bin/env python3
"""orze bug-fixer: continuously monitors an orze instance for stuck/zombie/deadlock
conditions and self-heals.

Platform-level tool — only diagnoses and fixes orze/farm.py bugs.
Never touches project-specific scripts (train scripts, eval scripts, etc.).

Reads orze.yaml for configuration. Runs forever alongside farm.py.

Usage:
    python orze/bug_fixer.py -c orze.yaml
"""

import argparse
import datetime
import json
import logging
import os
import re
import signal
import subprocess
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BUG-FIXER] %(levelname)s %(message)s",
)
logger = logging.getLogger("orze.bug_fixer")

# ─── Defaults (overridable via orze.yaml bug_fixer section) ──────────────────
DEFAULTS = {
    "check_interval": 60,
    "stale_training_min": 45,
    "stale_eval_min": 60,
    "heartbeat_timeout_min": 5,
    "max_fixes_per_hour": 3,
    "stale_claim_min": 120,
    "min_disk_gb": 50,
    "gpu_busy_threshold_mb": 5000,
}


def load_config(config_path: str) -> dict:
    """Load orze.yaml and extract bug_fixer settings."""
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    bf_cfg = cfg.get("bug_fixer", {})
    merged = {**DEFAULTS, **bf_cfg}
    merged["_orze"] = cfg
    return merged


def run_cmd(cmd, timeout=30):
    """Run a shell command and return (returncode, stdout, stderr)."""
    try:
        r = subprocess.run(
            cmd, shell=True, capture_output=True, text=True, timeout=timeout
        )
        return r.returncode, r.stdout.strip(), r.stderr.strip()
    except subprocess.TimeoutExpired:
        return -1, "", "timeout"
    except Exception as e:
        return -1, "", str(e)


# ─── Detectors (all generic — no project-specific logic) ─────────────────────

def check_orze_alive(cfg):
    """Check if the farm.py process is running and producing log output."""
    issues = []
    orze_cfg = cfg["_orze"]
    config_file = cfg.get("_config_file", "orze.yaml")

    rc, out, _ = run_cmd(f"pgrep -f 'farm.py.*{config_file}'")
    if rc != 0 or not out.strip():
        issues.append({
            "type": "orze_dead",
            "severity": "critical",
            "message": "Orze farm.py is not running!",
        })
        return issues

    # Check log freshness
    results_dir = Path(orze_cfg.get("results_dir", "results"))
    log_file = results_dir / "farm.log"
    if not log_file.exists():
        # Try orze.log as fallback
        log_file = results_dir / "orze.log"
    if log_file.exists():
        age_min = (time.time() - log_file.stat().st_mtime) / 60
        if age_min > cfg["heartbeat_timeout_min"]:
            issues.append({
                "type": "orze_stalled",
                "severity": "high",
                "message": f"Orze log not updated in {age_min:.1f} minutes",
                "details": str(log_file),
            })
    return issues


def check_zombie_processes(cfg):
    """Check for zombie (defunct) python processes."""
    issues = []
    rc, out, _ = run_cmd("ps aux | grep 'python' | grep '<defunct>' | grep -v grep")
    if rc == 0 and out.strip():
        zombies = [l for l in out.strip().split("\n") if l.strip()]
        if zombies:
            issues.append({
                "type": "zombie_processes",
                "severity": "medium",
                "message": f"Found {len(zombies)} zombie python processes",
                "details": "\n".join(zombies[:5]),
            })
    return issues


def check_stuck_processes(cfg):
    """Check for training/eval processes exceeding timeout."""
    issues = []
    orze_cfg = cfg["_orze"]
    train_script = Path(orze_cfg.get("train_script", "train.py")).name
    eval_script = Path(orze_cfg.get("eval_script", "")).name if orze_cfg.get("eval_script") else None

    # Check training
    rc, out, _ = run_cmd(
        f"ps -eo pid,etimes,args --sort=-etimes | grep '{train_script}' | grep -v grep"
    )
    if rc == 0 and out.strip():
        for line in out.strip().split("\n"):
            parts = line.split(None, 2)
            if len(parts) < 3:
                continue
            pid, elapsed_sec, cmd = parts[0], int(parts[1]), parts[2]
            elapsed_min = elapsed_sec / 60
            if elapsed_min > cfg["stale_training_min"]:
                rc2, cpu_out, _ = run_cmd(f"ps -p {pid} -o %cpu --no-headers")
                cpu_pct = float(cpu_out.strip()) if cpu_out.strip() else 0
                if cpu_pct < 5.0:
                    idea_match = re.search(r"idea-\S+", cmd)
                    idea_id = idea_match.group() if idea_match else "unknown"
                    issues.append({
                        "type": "stuck_training",
                        "severity": "high",
                        "message": f"Training {idea_id} stuck {elapsed_min:.0f}min (CPU={cpu_pct:.1f}%)",
                        "pid": pid,
                    })

    # Check evals
    if eval_script:
        rc, out, _ = run_cmd(
            f"ps -eo pid,etimes,args --sort=-etimes | grep '{eval_script}' | grep -v grep"
        )
        if rc == 0 and out.strip():
            for line in out.strip().split("\n"):
                parts = line.split(None, 2)
                if len(parts) < 3:
                    continue
                pid, elapsed_sec, cmd = parts[0], int(parts[1]), parts[2]
                elapsed_min = elapsed_sec / 60
                if elapsed_min > cfg["stale_eval_min"]:
                    idea_match = re.search(r"idea-\S+", cmd)
                    idea_id = idea_match.group() if idea_match else "unknown"
                    issues.append({
                        "type": "stuck_eval",
                        "severity": "high",
                        "message": f"Eval {idea_id} stuck {elapsed_min:.0f}min",
                        "pid": pid,
                    })
    return issues


def check_disk_space(cfg):
    """Check available disk space."""
    issues = []
    results_dir = Path(cfg["_orze"].get("results_dir", "results"))
    try:
        st = os.statvfs(str(results_dir))
        free_gb = (st.f_bavail * st.f_frsize) / (1024**3)
        if free_gb < cfg["min_disk_gb"]:
            issues.append({
                "type": "low_disk",
                "severity": "critical",
                "message": f"Only {free_gb:.1f}GB free (threshold: {cfg['min_disk_gb']}GB)",
            })
    except Exception:
        pass
    return issues


def check_stale_claims(cfg):
    """Check for ideas claimed but never completed."""
    issues = []
    results_dir = Path(cfg["_orze"].get("results_dir", "results"))
    now = time.time()
    threshold = cfg["stale_claim_min"] * 60
    try:
        for d in results_dir.iterdir():
            if not d.is_dir() or not d.name.startswith("idea-"):
                continue
            claim = d / "claim.json"
            metrics = d / "metrics.json"
            if claim.exists() and not metrics.exists():
                if (now - claim.stat().st_mtime) > threshold:
                    issues.append({
                        "type": "stale_claim",
                        "severity": "medium",
                        "message": f"{d.name} claimed {(now - claim.stat().st_mtime)/60:.0f}min ago, no metrics",
                    })
    except Exception:
        pass
    return issues


def check_orze_errors(cfg):
    """Scan orze log for errors originating in farm.py (not subprocess scripts)."""
    issues = []
    results_dir = Path(cfg["_orze"].get("results_dir", "results"))
    log_file = results_dir / "farm.log"
    if not log_file.exists():
        log_file = results_dir / "orze.log"
    if not log_file.exists():
        return issues

    # Only look for errors in farm.py itself, not in subprocess output
    rc, out, _ = run_cmd(
        f"tail -300 {log_file} | grep -E '\\[ERROR\\]|\\[CRITICAL\\]|Traceback.*farm\\.py' | tail -10"
    )
    if rc == 0 and out.strip():
        seen = set()
        for line in out.strip().split("\n"):
            err_match = re.search(
                r"(KeyError|TypeError|AttributeError|RuntimeError|"
                r"ValueError|OSError|FileNotFoundError|deadlock|"
                r"CRITICAL|Traceback)", line, re.I
            )
            if err_match:
                err_type = err_match.group(1)
                if err_type not in seen:
                    seen.add(err_type)
                    issues.append({
                        "type": "orze_error",
                        "severity": "high",
                        "message": f"farm.py error: {err_type}",
                        "details": line[:300],
                    })
    return issues


# ─── Actions ─────────────────────────────────────────────────────────────────

_fix_timestamps: list = []
_fixed_issues: set = set()


def issue_hash(issue):
    return f"{issue['type']}:{issue.get('message', '')[:60]}"


def should_fix(issue, cfg):
    h = issue_hash(issue)
    if h in _fixed_issues:
        return False
    if issue["severity"] not in ("critical", "high"):
        return False
    now = time.time()
    recent = [t for t in _fix_timestamps if now - t < 3600]
    return len(recent) < cfg["max_fixes_per_hour"]


def save_issue(issue, cfg):
    issues_dir = Path(cfg["_orze"].get("results_dir", "results")) / "bug_fixer_issues"
    issues_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    p = issues_dir / f"{ts}_{issue['type']}.json"
    p.write_text(json.dumps(issue, indent=2, default=str))
    return p


def kill_process(pid):
    logger.info("Killing stuck PID=%s", pid)
    run_cmd(f"kill {pid}")
    time.sleep(2)
    rc, _, _ = run_cmd(f"ps -p {pid} -o pid= 2>/dev/null")
    if rc != 0:
        logger.info("PID=%s terminated", pid)
        return True
    run_cmd(f"kill -9 {pid}")
    time.sleep(1)
    return True


def restart_orze(cfg):
    """Restart the orze farm.py process."""
    config_file = cfg.get("_config_file", "orze.yaml")
    python = cfg["_orze"].get("python", sys.executable)
    project_dir = cfg.get("_project_dir", ".")
    results_dir = Path(cfg["_orze"].get("results_dir", "results"))
    log_file = results_dir / "farm.log"

    logger.info("Restarting Orze...")
    run_cmd(f"pkill -f 'farm.py.*{config_file}'")
    time.sleep(5)
    run_cmd(f"pkill -9 -f 'farm.py.*{config_file}'")
    time.sleep(2)

    orze_script = Path(project_dir) / "orze" / "farm.py"
    run_cmd(
        f"cd {project_dir} && nohup {python} {orze_script} -c {config_file} "
        f">> {log_file} 2>&1 &",
        timeout=10,
    )
    time.sleep(3)
    rc, pid_out, _ = run_cmd(f"pgrep -f 'farm.py.*{config_file}'")
    if pid_out.strip():
        logger.info("Orze restarted, PID=%s", pid_out.strip())
        return True
    logger.error("Failed to restart Orze!")
    return False


def spawn_claude_fix(issue, cfg):
    """Spawn Claude CLI to diagnose and fix a farm.py bug."""
    h = issue_hash(issue)
    save_issue(issue, cfg)

    results_dir = Path(cfg["_orze"].get("results_dir", "results"))
    log_file = results_dir / "farm.log"
    if not log_file.exists():
        log_file = results_dir / "orze.log"
    orze_dir = cfg.get("_orze_dir", "orze")

    rc, log_tail, _ = run_cmd(f"tail -200 {log_file}")

    prompt = f"""You are the Orze platform bug-fixer. An issue was detected in the running Orze system.

## Issue
- **Type**: {issue['type']}
- **Severity**: {issue['severity']}
- **Message**: {issue['message']}
- **Details**: {issue.get('details', 'N/A')}

## Recent Orze Log
```
{log_tail[-3000:]}
```

## Your Task
1. Read `{orze_dir}/farm.py` and diagnose the root cause
2. If it's a bug in farm.py, fix it with a minimal, targeted patch
3. If it's an operational issue (stuck process, resource exhaustion), report it but do NOT change code
4. If the error comes from a user's training/eval script (not farm.py), report "Not an orze bug" and skip

## CRITICAL RULES
- You may ONLY modify `{orze_dir}/farm.py` — this is the orze platform
- NEVER modify project scripts (train scripts, eval scripts, dataset loaders, etc.)
- NEVER modify orze.yaml, ideas.md, RESEARCH_RULES.md, or any user files
- Keep fixes minimal — only fix the specific bug, don't refactor
- Verify syntax: python3 -c "import ast; ast.parse(open('{orze_dir}/farm.py').read())"
- Commit locally: cd {orze_dir} && git add farm.py && git commit -m "fix(auto): <description>"
- Do NOT push — local commits only, human will review and push
- If this is NOT a farm.py bug, just say so and exit without changes
"""

    logger.info("Spawning Claude to diagnose: %s", issue["message"])
    issues_dir = results_dir / "bug_fixer_issues"
    issues_dir.mkdir(parents=True, exist_ok=True)

    try:
        env = os.environ.copy()
        env.pop("CLAUDECODE", None)
        env.pop("CLAUDE_CODE_ENTRYPOINT", None)
        result = subprocess.run(
            ["claude", "-p", prompt, "--dangerously-skip-permissions",
             "--output-format", "text", "--model", "sonnet"],
            capture_output=True, text=True, timeout=300,
            cwd=cfg.get("_project_dir", "."),
            env=env,
        )
        response = result.stdout[-5000:] if result.stdout else ""
        logger.info("Claude response:\n%s", response[:2000])

        resp_file = issues_dir / f"response_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        resp_file.write_text(response)

        _fixed_issues.add(h)
        _fix_timestamps.append(time.time())
        return True
    except subprocess.TimeoutExpired:
        logger.warning("Claude session timed out for: %s", issue["message"])
        return False
    except FileNotFoundError:
        logger.warning("Claude CLI not found — skipping automated fix")
        return False
    except Exception as e:
        logger.error("Failed to spawn Claude: %s", e)
        return False


# ─── Main Loop ───────────────────────────────────────────────────────────────

def run_all_checks(cfg):
    issues = []
    issues.extend(check_orze_alive(cfg))
    issues.extend(check_zombie_processes(cfg))
    issues.extend(check_stuck_processes(cfg))
    issues.extend(check_disk_space(cfg))
    issues.extend(check_stale_claims(cfg))
    issues.extend(check_orze_errors(cfg))
    return issues


def handle_issue(issue, cfg):
    itype = issue["type"]

    if itype in ("stuck_training", "stuck_eval") and issue.get("pid"):
        kill_process(issue["pid"])
        save_issue({**issue, "action": "auto_killed"}, cfg)
        return

    if itype == "orze_dead":
        restart_orze(cfg)
        save_issue({**issue, "action": "auto_restarted"}, cfg)
        return

    if itype in ("orze_error", "orze_stalled") and should_fix(issue, cfg):
        spawn_claude_fix(issue, cfg)
        return

    save_issue({**issue, "action": "logged_only"}, cfg)


def main():
    parser = argparse.ArgumentParser(description="Orze bug-fixer agent")
    parser.add_argument("-c", "--config", default="orze.yaml", help="Path to orze.yaml")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    project_dir = config_path.parent
    orze_dir = project_dir / "orze"

    cfg = load_config(str(config_path))
    cfg["_config_file"] = config_path.name
    cfg["_project_dir"] = str(project_dir)
    cfg["_orze_dir"] = str(orze_dir)

    interval = cfg["check_interval"]

    logger.info("=" * 60)
    logger.info("Orze bug-fixer started")
    logger.info("Config: %s", config_path)
    logger.info("Project: %s", project_dir)
    logger.info("Check interval: %ds", interval)
    logger.info("=" * 60)

    cycle = 0
    while True:
        try:
            cycle += 1
            issues = run_all_checks(cfg)

            if issues:
                logger.info(
                    "[Cycle %d] Found %d issues: %s",
                    cycle, len(issues),
                    ", ".join(i["type"] for i in issues),
                )
                for issue in issues:
                    handle_issue(issue, cfg)
            elif cycle % 30 == 0:
                logger.info("[Cycle %d] All clear", cycle)

        except Exception as e:
            logger.error("Check cycle failed: %s", e, exc_info=True)

        time.sleep(interval)


if __name__ == "__main__":
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    main()
