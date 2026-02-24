#!/usr/bin/env python3
"""
orze: GPU experiment orchestrator using filesystem coordination.

Parses ideas from a markdown file, claims them via atomic mkdir,
launches training on free GPUs, monitors health, runs post-training
evaluation, and generates a configurable leaderboard.

Configuration via orze.yaml (optional — works without it).

Usage:
    python farm.py                              # all GPUs, continuous
    python farm.py -c orze.yaml --gpus 0,1      # with project config
    python farm.py --once                       # one cycle then exit
    python farm.py --report-only                # just regenerate report
    python farm.py --role-only research         # run one role once
"""

import argparse
import datetime
import html as html_mod
import json
import logging
import os
import platform
import re
import shutil
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import copy

import atexit

import yaml

__version__ = "1.10.1"

logger = logging.getLogger("orze")


# ---------------------------------------------------------------------------
# Process group helpers — ensures all child processes are killed on shutdown
# ---------------------------------------------------------------------------

def _new_process_group():
    """preexec_fn for Popen: put subprocess in its own process group."""
    os.setsid()


def _kill_pg(proc: subprocess.Popen, sig=signal.SIGTERM):
    """Send a signal to the entire process group of *proc*.

    Falls back to sending directly to the process if the group
    lookup fails (e.g. process already dead).
    """
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, sig)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.send_signal(sig)
        except (ProcessLookupError, OSError):
            pass


def _terminate_and_reap(proc: subprocess.Popen, label: str = "",
                        timeout: float = 10):
    """SIGTERM the process group, wait, then SIGKILL if needed."""
    _kill_pg(proc, signal.SIGTERM)
    try:
        proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        logger.warning("Force killing %s (PID %d)", label or "process", proc.pid)
        _kill_pg(proc, signal.SIGKILL)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            logger.error("Failed to reap %s (PID %d) after SIGKILL",
                         label or "process", proc.pid)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def atomic_write(path: Path, content: str):
    """Write content atomically via tmp+replace (safe across machines)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_host = "".join(c if c.isalnum() else "_" for c in socket.gethostname())
    tmp = path.with_name(f"{path.name}.{safe_host}.{os.getpid()}.tmp")
    tmp.write_text(content, encoding="utf-8")
    tmp.replace(path)


def _get_checkpoint_dir(cfg: dict) -> Optional[Path]:
    """Extract --checkpoint-dir from train_extra_args."""
    args = cfg.get("train_extra_args") or []
    for i, arg in enumerate(args):
        if str(arg) == "--checkpoint-dir" and i + 1 < len(args):
            return Path(str(args[i + 1]))
    return None


def tail_file(path: Path, n_bytes: int = 4096) -> str:
    """Read last n_bytes of a file."""
    try:
        size = path.stat().st_size
        with open(path, "rb") as f:
            f.seek(max(0, size - n_bytes))
            return f.read().decode("utf-8", errors="replace")
    except Exception:
        return ""


def _format_args(args, template_vars: dict) -> list:
    """Safely format arguments without crashing on literal {} braces."""
    if args is None:
        args = []
    elif isinstance(args, str):
        args = [args]
    elif not isinstance(args, list):
        try:
            args = list(args)
        except TypeError:
            args = [args]
    formatted = []
    for arg in args:
        s = str(arg)
        for k, v in template_vars.items():
            s = s.replace(f"{{{k}}}", str(v))
        formatted.append(s)
    return formatted


def deep_get(obj: dict, dotpath: str, default=None):
    """Get nested dict value by dot path: 'a.b.c' -> obj[a][b][c]."""
    keys = dotpath.split(".")
    for k in keys:
        if isinstance(obj, dict):
            obj = obj.get(k, default)
        else:
            return default
    return obj


def _resolve_primary_metric(eval_data: dict, cfg: dict,
                            eval_file: str) -> Any:
    """Extract the primary metric value from eval report data.

    Looks up the primary_metric key in report.columns to find the
    source path, then extracts the value from eval_data.
    """
    primary = cfg.get("report", {}).get("primary_metric", "test_accuracy")
    columns = cfg.get("report", {}).get("columns", [])
    for col in columns:
        if col.get("key") == primary:
            src = col.get("source", "")
            if ":" in src:
                src_file, json_path = src.split(":", 1)
                if src_file == eval_file:
                    return deep_get(eval_data, json_path)
    # Fallback: try metrics.{primary} directly
    return deep_get(eval_data, f"metrics.{primary}")


# ---------------------------------------------------------------------------
# Filesystem locks (mkdir-based, works across machines on shared FS)
# ---------------------------------------------------------------------------

def _fs_lock(lock_dir: Path, stale_seconds: float = 600) -> bool:
    """Acquire a filesystem lock via mkdir. Returns True if acquired.
    Stale locks (older than stale_seconds) are broken automatically."""
    try:
        lock_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        # Check if stale
        try:
            lock_file = lock_dir / "lock.json"
            if lock_file.exists():
                age = time.time() - lock_file.stat().st_mtime
            else:
                # lock.json missing (crash between mkdir and write) — use dir mtime
                age = time.time() - lock_dir.stat().st_mtime

            if age > stale_seconds:
                try:
                    shutil.rmtree(lock_dir)
                except OSError:
                    pass  # Another node already deleted the stale lock
                try:
                    lock_dir.mkdir(parents=True, exist_ok=False)
                except FileExistsError:
                    return False
            else:
                return False
        except Exception:
            return False

    # Write lock info
    lock_info = {
        "host": socket.gethostname(),
        "pid": os.getpid(),
        "time": datetime.datetime.now().isoformat(),
    }
    try:
        atomic_write(lock_dir / "lock.json", json.dumps(lock_info))
    except Exception:
        pass
    return True


def _fs_unlock(lock_dir: Path):
    """Release a filesystem lock."""
    try:
        shutil.rmtree(lock_dir)
    except Exception:
        pass


# ---------------------------------------------------------------------------
# Project config (orze.yaml)
# ---------------------------------------------------------------------------

DEFAULT_CONFIG = {
    "train_script": "train_idea.py",
    "ideas_file": "ideas.md",
    "base_config": "configs/base.yaml",
    "results_dir": "results",
    "python": sys.executable,
    "train_extra_args": [],
    "train_extra_env": {},
    "timeout": 3600,
    "poll": 30,
    "gpu_mem_threshold": 2000,
    "pre_script": None,
    "pre_args": [],
    "pre_timeout": 3600,
    "eval_script": None,
    "eval_args": [],
    "eval_timeout": 3600,
    "eval_output": "eval_report.json",
    "post_scripts": [],
    "cleanup": {
        "script": None,
        "interval": 100,
        "patterns": [],
    },
    "report": {
        "title": "Orze Report",
        "primary_metric": "test_accuracy",
        "sort": "descending",
        "columns": [
            {"key": "test_accuracy", "label": "Accuracy", "fmt": ".4f"},
            {"key": "test_loss", "label": "Loss", "fmt": ".4f"},
            {"key": "training_time", "label": "Time(s)", "fmt": ".0f"},
        ],
    },
    "stall_minutes": 0,         # 0 = disabled
    "max_idea_failures": 0,     # 0 = disabled (never skip)
    "max_fix_attempts": 0,      # 0 = disabled; executor LLM fix attempts per idea
    "min_disk_gb": 0,           # 0 = disabled
    "orphan_timeout_hours": 0,  # 0 = disabled
    "roles": {},
    "notifications": {
        "enabled": False,
        "on": ["completed", "failed", "new_best"],
        "channels": [],
    },
}


def load_project_config(path: Optional[str]) -> dict:
    """Load orze.yaml and merge with defaults. Returns full config dict."""
    cfg = dict(DEFAULT_CONFIG)
    cfg["report"] = dict(DEFAULT_CONFIG["report"])

    if path and Path(path).exists():
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for k, v in raw.items():
            if k == "report" and isinstance(v, dict):
                cfg["report"] = {**cfg["report"], **v}
            else:
                cfg[k] = v
        # Fix YAML 'on:' boolean parsing — YAML interprets 'on' as True
        # so notifications.on becomes notifications[True] instead of notifications["on"]
        ncfg = cfg.get("notifications")
        if isinstance(ncfg, dict) and True in ncfg and "on" not in ncfg:
            ncfg["on"] = ncfg.pop(True)
            logger.info("Fixed YAML 'on:' boolean key in notifications config")

        logger.info("Loaded config from %s", path)
    elif path:
        logger.warning("Config file %s not found, using defaults", path)

    # Migrate legacy research: into roles: dict
    if "research" in cfg and isinstance(cfg["research"], dict):
        logger.warning("Migrating legacy 'research:' config to 'roles: {research: ...}'. "
                        "Update orze.yaml to use the 'roles:' format directly.")
        if not cfg.get("roles"):
            cfg["roles"] = {"research": cfg["research"]}
        elif "research" not in cfg["roles"]:
            cfg["roles"]["research"] = cfg["research"]

    # Auto-discover research backends from environment API keys.
    # Only activates if no research roles are explicitly configured.
    roles = cfg.get("roles") or {}
    has_research_role = any(
        isinstance(rc, dict) and rc.get("mode") in ("claude", "research")
        for rc in roles.values()
    )
    if not has_research_role:
        _AUTO_BACKENDS = [
            ("GEMINI_API_KEY", "gemini", "gemini-2.5-flash"),
            ("OPENAI_API_KEY", "openai", "gpt-4o"),
            ("ANTHROPIC_API_KEY", "anthropic", None),
        ]
        discovered = []
        for env_var, backend, default_model in _AUTO_BACKENDS:
            if os.environ.get(env_var):
                role_name = f"research_{backend}"
                role_cfg = {"mode": "research", "backend": backend}
                if default_model:
                    role_cfg["model"] = default_model
                if "roles" not in cfg:
                    cfg["roles"] = {}
                cfg["roles"][role_name] = role_cfg
                discovered.append(f"{backend} ({env_var})")
        if discovered:
            logger.info("Auto-discovered research backends: %s",
                        ", ".join(discovered))

    return cfg


def _validate_config(cfg: dict) -> list:
    """Validate orze config on startup. Returns list of error strings."""
    errors = []

    # Validate roles
    roles = cfg.get("roles")
    if roles and isinstance(roles, dict):
        for rname, rcfg in roles.items():
            if not isinstance(rcfg, dict):
                errors.append(f"roles.{rname}: expected dict, got {type(rcfg).__name__}")
                continue
            mode = rcfg.get("mode", "script")
            if mode not in ("script", "claude", "research"):
                errors.append(f"roles.{rname}.mode: '{mode}' is invalid "
                              f"(expected 'script', 'claude', or 'research')")
            if mode == "claude" and not rcfg.get("rules_file"):
                errors.append(f"roles.{rname}: mode 'claude' requires 'rules_file'")
            if mode == "script" and not rcfg.get("script"):
                errors.append(f"roles.{rname}: mode 'script' requires 'script'")
            if mode == "research" and not rcfg.get("backend"):
                errors.append(f"roles.{rname}: mode 'research' requires 'backend' "
                              f"(gemini, openai, anthropic, ollama, custom)")

    # Validate numeric fields
    for key in ("timeout", "poll", "eval_timeout", "stall_minutes"):
        val = cfg.get(key)
        if val is not None and (not isinstance(val, (int, float)) or val <= 0):
            errors.append(f"{key}: must be a positive number, got {val!r}")

    # Validate eval config consistency
    if cfg.get("eval_script") and not cfg.get("eval_output"):
        errors.append("eval_script is set but eval_output is missing")

    return errors


# ---------------------------------------------------------------------------
# Ideas parsing
# ---------------------------------------------------------------------------

PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


_parse_ideas_cache: Dict[str, Any] = {"mtime": 0.0, "result": {}, "path": ""}


def parse_ideas(path: str) -> Dict[str, dict]:
    """Parse ideas.md into {idea_id: {title, priority, config, raw}}.

    Results are cached by file mtime to avoid re-parsing on every iteration.
    """
    try:
        p = Path(path)
        mtime = p.stat().st_mtime
        if (mtime == _parse_ideas_cache["mtime"]
                and path == _parse_ideas_cache["path"]
                and _parse_ideas_cache["result"]):
            return _parse_ideas_cache["result"]
        text = p.read_text(encoding="utf-8")
    except OSError:
        return {}
    ideas = {}
    pattern = re.compile(r"^## (idea-\d+):\s*(.+?)$", re.MULTILINE)
    matches = list(pattern.finditer(text))

    for i, m in enumerate(matches):
        idea_id = m.group(1)
        title = m.group(2).strip()
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        raw = text[start:end]

        pri_match = re.search(r"\*\*Priority\*\*:\s*(\w+)", raw)
        priority = pri_match.group(1).lower() if pri_match else "medium"

        yaml_match = re.search(r"```ya?ml\s*\n(.*?)```", raw, re.DOTALL)
        config = {}
        if yaml_match:
            try:
                config = yaml.safe_load(yaml_match.group(1)) or {}
            except yaml.YAMLError:
                pass

        # Sanitize config: replace non-numeric values in numeric fields
        config = _sanitize_config(config)

        ideas[idea_id] = {
            "title": title,
            "priority": priority,
            "config": config,
            "raw": raw.strip(),
        }
    _parse_ideas_cache["mtime"] = mtime
    _parse_ideas_cache["path"] = path
    _parse_ideas_cache["result"] = ideas
    return ideas


def _sanitize_config(config: dict) -> dict:
    """Sanitize config by replacing invalid numeric values with defaults.

    Handles cases where AI-generated ideas use strings like 'variable' or 'auto'
    in fields that expect integers (e.g., sequence_length, max_frames).
    """
    if not isinstance(config, dict):
        return config

    # Known numeric fields that should be integers
    numeric_fields = {
        ("training", "sequence_length"): 32,
        ("training", "batch_size"): 16,
        ("training", "epochs"): 10,
        ("data", "batch_size"): 16,
        ("data", "frame_sampling", "max_frames"): 32,
        ("optimizer", "max_epochs"): 10,
    }

    # Known intermediate fields that must be dicts (not lists/scalars)
    dict_fields = {
        ("data", "frame_sampling"): {},
    }

    def sanitize_value(value, default):
        """Try to convert to int; if it fails, return default."""
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                logger.warning(f"Replacing invalid numeric value '{value}' with {default}")
                return default
        return value

    # Deep copy to avoid modifying original
    config = copy.deepcopy(config)

    # Ensure known intermediate fields are dicts
    for path, default in dict_fields.items():
        current = config
        for key in path[:-1]:
            if key not in current or not isinstance(current[key], dict):
                break
            current = current[key]
        else:
            final_key = path[-1]
            if final_key in current and not isinstance(current[final_key], dict):
                logger.warning(
                    f"Replacing non-dict value for '{'.'.join(path)}' "
                    f"(was {type(current[final_key]).__name__}) with {default}"
                )
                current[final_key] = default

    # Sanitize known numeric fields
    for path, default in numeric_fields.items():
        current = config
        for i, key in enumerate(path[:-1]):
            if key not in current or not isinstance(current[key], dict):
                break
            current = current[key]
        else:
            # We successfully navigated to the parent dict
            final_key = path[-1]
            if final_key in current:
                current[final_key] = sanitize_value(current[final_key], default)

    return config


# ---------------------------------------------------------------------------
# HP Sweep expansion
# ---------------------------------------------------------------------------

# Keys whose list values are structural (not sweep candidates)
_SWEEP_BLOCKLIST = frozenset({
    "betas", "split_ratio", "stack_layers", "stack_dims",
    "downsampling_factors", "num_heads", "feedforward_dim",
    "output_range", "augmentations", "transforms",
})


def _find_sweep_keys(config: dict, prefix: str = "") -> Dict[str, list]:
    """Recursively find list-valued scalar keys suitable for sweeping.

    Returns dict of dotpath -> list-of-values.
    """
    found = {}
    for key, val in config.items():
        dotpath = f"{prefix}.{key}" if prefix else key
        if isinstance(val, dict):
            found.update(_find_sweep_keys(val, dotpath))
        elif isinstance(val, list) and key not in _SWEEP_BLOCKLIST:
            # Only sweep if all elements are scalars (int, float, str, bool)
            if val and all(isinstance(v, (int, float, str, bool)) for v in val):
                found[dotpath] = val
    return found


def _set_nested(config: dict, dotpath: str, value):
    """Set a value at a dot-separated path in a nested dict."""
    keys = dotpath.split(".")
    d = config
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def expand_sweeps(ideas: Dict[str, dict],
                  max_combos: int = 20) -> Dict[str, dict]:
    """Expand ideas with list-valued HPs into sub-run entries.

    Sub-run IDs use format: idea-123~key=val~key2=val2
    Original ideas with no sweepable keys pass through unchanged.
    Ideas whose ID already contains '~' are skipped (no double-expansion).
    """
    import itertools

    expanded = {}
    for idea_id, idea in ideas.items():
        # Skip already-expanded sub-runs
        if "~" in idea_id:
            expanded[idea_id] = idea
            continue

        sweep_keys = _find_sweep_keys(idea.get("config", {}))
        if not sweep_keys:
            expanded[idea_id] = idea
            continue

        # Compute Cartesian product
        keys = sorted(sweep_keys.keys())
        values = [sweep_keys[k] for k in keys]
        combos = list(itertools.product(*values))

        if len(combos) > max_combos:
            logger.warning(
                "Sweep for %s has %d combos (max %d), skipping expansion",
                idea_id, len(combos), max_combos)
            expanded[idea_id] = idea
            continue

        logger.info("Expanding %s into %d sweep sub-runs (%s)",
                    idea_id, len(combos),
                    ", ".join(f"{k}={len(v)}" for k, v in zip(keys, values)))

        for combo in combos:
            suffix_parts = []
            sub_config = copy.deepcopy(idea["config"])
            for k, v in zip(keys, combo):
                short_key = k.split(".")[-1]
                suffix_parts.append(f"{short_key}={v}")
                _set_nested(sub_config, k, v)

            sub_id = f"{idea_id}~{'~'.join(suffix_parts)}"
            expanded[sub_id] = {
                "title": idea["title"],
                "priority": idea.get("priority", "medium"),
                "config": sub_config,
                "raw": idea.get("raw", ""),
                "_sweep_parent": idea_id,
            }

    return expanded


def get_unclaimed(ideas: Dict[str, dict], results_dir: Path,
                  skipped: Optional[set] = None) -> List[str]:
    """Return idea IDs with no results dir, sorted by priority then ID.
    Excludes ideas in the skipped set."""
    unclaimed = []
    for idea_id in ideas:
        if skipped and idea_id in skipped:
            continue
        if not (results_dir / idea_id).exists():
            unclaimed.append(idea_id)

    def sort_key(idea_id):
        pri = PRIORITY_ORDER.get(ideas[idea_id]["priority"], 2)
        match = re.search(r"\d+", idea_id)
        return (pri, int(match.group()) if match else 999999)

    unclaimed.sort(key=sort_key)
    return unclaimed


# ---------------------------------------------------------------------------
# Claiming (atomic mkdir)
# ---------------------------------------------------------------------------

def claim(idea_id: str, results_dir: Path, gpu: int) -> bool:
    """Atomically claim an idea via mkdir. Returns True if we got it."""
    idea_dir = results_dir / idea_id
    try:
        idea_dir.mkdir(parents=True, exist_ok=False)
    except FileExistsError:
        return False

    claim_info = {
        "claimed_by": socket.gethostname(),
        "claimed_at": datetime.datetime.now().isoformat(),
        "pid": os.getpid(),
        "gpu": gpu,
    }
    atomic_write(idea_dir / "claim.json", json.dumps(claim_info, indent=2))
    return True


# ---------------------------------------------------------------------------
# GPU management
# ---------------------------------------------------------------------------

def get_gpu_memory_used(gpu_id: int) -> Optional[int]:
    """Get GPU memory used in MiB. Returns None on failure."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used",
             "--format=csv,noheader,nounits", f"--id={gpu_id}"],
            capture_output=True, text=True, timeout=10,
        )
        return int(result.stdout.strip())
    except Exception:
        return None


def _eval_already_running(idea_id: str, cfg: dict = None) -> bool:
    """Check if an eval process is already running for this idea."""
    eval_script = "eval"
    if cfg:
        script_path = cfg.get("eval_script", "")
        if script_path:
            eval_script = Path(script_path).stem
    try:
        result = subprocess.run(
            ["pgrep", "-f", f"{eval_script}.*{idea_id}"],
            capture_output=True, text=True, timeout=5,
        )
        return result.returncode == 0
    except Exception:
        return False


def detect_all_gpus() -> List[int]:
    """Auto-detect available GPU indices."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return [int(x.strip()) for x in result.stdout.strip().split("\n")
                if x.strip()]
    except Exception:
        return []


def get_free_gpus(gpu_ids: List[int], active: Dict[int, "TrainingProcess"],
                  mem_threshold: int = 2000) -> List[int]:
    """Return GPUs not in active dict and with low memory usage."""
    free = []
    for g in gpu_ids:
        if g in active:
            continue
        mem_used = get_gpu_memory_used(g)
        if mem_used is not None and mem_used > mem_threshold:
            continue
        free.append(g)
    return free


# ---------------------------------------------------------------------------
# Training process management
# ---------------------------------------------------------------------------

@dataclass
class TrainingProcess:
    idea_id: str
    gpu: int
    process: subprocess.Popen
    start_time: float
    log_path: Path
    timeout: float
    _log_fh: Any = field(default=None, repr=False)
    _last_log_size: int = field(default=0, repr=False)
    _last_log_check: float = field(default=0.0, repr=False)
    _stall_since: float = field(default=0.0, repr=False)

    def close_log(self):
        """Close the log file handle if open."""
        if self._log_fh and not self._log_fh.closed:
            try:
                self._log_fh.close()
            except Exception:
                pass


@dataclass
class EvalProcess:
    """Tracks a non-blocking eval subprocess."""
    idea_id: str
    gpu: int
    process: subprocess.Popen
    start_time: float
    log_path: Path
    timeout: float
    _log_fh: Any = field(default=None, repr=False)

    def close_log(self):
        if self._log_fh and not self._log_fh.closed:
            try:
                self._log_fh.close()
            except Exception:
                pass


@dataclass
class RoleProcess:
    """Tracks a non-blocking agent role subprocess."""
    role_name: str
    process: subprocess.Popen
    start_time: float
    log_path: Path
    timeout: float
    lock_dir: Path
    cycle_num: int
    _log_fh: Any = field(default=None, repr=False)
    ideas_pre_size: int = 0  # ideas.md size before role started
    ideas_pre_count: int = 0  # idea count before role started

    def close_log(self):
        if self._log_fh and not self._log_fh.closed:
            try:
                self._log_fh.close()
            except Exception:
                pass


def run_pre_script(idea_id: str, gpu: int, cfg: dict) -> bool:
    """Run pre-training script if configured. Returns True if OK to proceed."""
    pre_script = cfg.get("pre_script")
    if not pre_script:
        return True

    python = cfg.get("python", sys.executable)
    pre_args = cfg.get("pre_args") or []
    pre_timeout = cfg.get("pre_timeout", 3600)

    cmd = [python, pre_script]
    cmd.extend(_format_args(pre_args, {"idea_id": idea_id, "gpu": gpu}))

    env = os.environ.copy()
    for k, v in (cfg.get("train_extra_env") or {}).items():
        env[k] = str(v)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    logger.info("Running pre-script for %s on GPU %d", idea_id, gpu)
    try:
        result = subprocess.run(
            cmd, env=env, capture_output=True, text=True,
            timeout=pre_timeout,
        )
        if result.returncode == 0:
            logger.info("Pre-script OK for %s", idea_id)
            return True
        else:
            logger.warning("Pre-script failed for %s (exit %d): %s",
                           idea_id, result.returncode,
                           result.stderr[-200:] if result.stderr else "")
            return False
    except subprocess.TimeoutExpired:
        logger.warning("Pre-script timed out for %s after %ds",
                       idea_id, pre_timeout)
        return False
    except Exception as e:
        logger.warning("Pre-script error for %s: %s", idea_id, e)
        return False


def launch(idea_id: str, gpu: int, results_dir: Path, cfg: dict) -> TrainingProcess:
    """Launch a training subprocess on the given GPU."""
    log_path = results_dir / idea_id / "train_output.log"

    python = cfg.get("python", sys.executable)
    train_script = cfg["train_script"]

    cmd = [
        python, train_script,
        "--idea-id", idea_id,
        "--results-dir", str(results_dir),
        "--ideas-md", cfg["ideas_file"],
        "--config", cfg["base_config"],
    ]
    for arg in (cfg.get("train_extra_args") or []):
        cmd.append(str(arg))

    env = os.environ.copy()
    for k, v in (cfg.get("train_extra_env") or {}).items():
        env[k] = str(v)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    # Keep file handle open for subprocess lifetime
    log_fh = open(log_path, "w", encoding="utf-8")
    try:
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
            preexec_fn=_new_process_group,
        )
    except Exception:
        log_fh.close()
        raise

    now = time.time()
    return TrainingProcess(
        idea_id=idea_id, gpu=gpu, process=proc,
        start_time=now, log_path=log_path,
        timeout=cfg.get("timeout", 3600),
        _log_fh=log_fh, _last_log_size=0,
        _last_log_check=now, _stall_since=0.0,
    )


def _write_failure(idea_dir: Path, reason: str):
    """Write a failure metrics.json atomically."""
    metrics = {
        "status": "FAILED",
        "error": reason,
        "timestamp": datetime.datetime.now().isoformat(),
    }
    atomic_write(idea_dir / "metrics.json", json.dumps(metrics, indent=2))


# ---------------------------------------------------------------------------
# Health monitoring
# ---------------------------------------------------------------------------

def check_stalled(tp: TrainingProcess, stall_minutes: int) -> bool:
    """Check if a process has stalled (no log growth). Returns True if stalled."""
    if stall_minutes <= 0:
        return False

    now = time.time()
    try:
        current_size = tp.log_path.stat().st_size
    except OSError:
        return False

    if current_size > tp._last_log_size:
        tp._last_log_size = current_size
        tp._last_log_check = now
        tp._stall_since = 0.0
        return False

    if tp._stall_since == 0.0:
        tp._stall_since = now
    elif (now - tp._stall_since) > stall_minutes * 60:
        return True

    return False


def detect_fatal_in_log(tp: TrainingProcess) -> Optional[str]:
    """Check log tail for fatal errors in a still-running process.

    Returns the matched error snippet if found, else None.
    The process may hang after printing a fatal error (e.g. OOM, NCCL
    timeout, segfault message) without exiting.  Rather than hardcoding
    a fixed pattern list, we look for any Python exception that was
    printed but the process is still alive — a sign it's hung.
    """
    tail = tail_file(tp.log_path, 8192)
    if not tail:
        return None
    # Look for a Python traceback followed by no further progress
    # (the traceback itself is in the last 8KB of the log)
    tb_idx = tail.rfind("Traceback (most recent call last)")
    if tb_idx == -1:
        return None
    # Extract from the traceback to the end
    snippet = tail[tb_idx:]
    # Only flag if there's very little output after the traceback
    # (< 200 chars — just the error message, no further training output)
    lines_after_tb = snippet.split("\n")
    if len(lines_after_tb) > 15:
        # Lots of output after the traceback — process recovered
        return None
    return snippet[:500]


def check_disk_space(path: Path, min_gb: float) -> bool:
    """Return True if disk has at least min_gb free. False if low."""
    if min_gb <= 0:
        return True
    try:
        usage = shutil.disk_usage(path)
        free_gb = usage.free / (1024 ** 3)
        return free_gb >= min_gb
    except Exception:
        return True


def cleanup_orphans(results_dir: Path, hours: float) -> int:
    """Remove result dirs with claim.json but no metrics.json older than hours.
    Returns count of cleaned dirs."""
    if hours <= 0:
        return 0

    cleaned = 0
    cutoff = time.time() - hours * 3600

    for d in results_dir.iterdir():
        if not d.is_dir() or not d.name.startswith("idea-"):
            continue
        claim_path = d / "claim.json"
        metrics_path = d / "metrics.json"

        if claim_path.exists() and not metrics_path.exists():
            try:
                last_activity = claim_path.stat().st_mtime
                log_path = d / "train_output.log"
                if log_path.exists():
                    last_activity = max(last_activity,
                                        log_path.stat().st_mtime)
                if last_activity < cutoff:
                    shutil.rmtree(d)
                    logger.info("Cleaned orphan: %s (last activity %.1fh ago)",
                                d.name, (time.time() - last_activity) / 3600)
                    cleaned += 1
            except Exception as e:
                logger.warning("Failed to clean orphan %s: %s", d.name, e)

    return cleaned


# ---------------------------------------------------------------------------
# Process checking (with health monitoring)
# ---------------------------------------------------------------------------

def _adaptive_stall_minutes(results_dir: Path, configured: int) -> int:
    """Compute adaptive stall timeout: min(configured, max(5, 2x median training time)).
    Falls back to configured value if not enough data."""
    if configured <= 0:
        return 0
    times = []
    try:
        for d in results_dir.iterdir():
            if not d.is_dir() or not d.name.startswith("idea-"):
                continue
            mp = d / "metrics.json"
            if not mp.exists():
                continue
            m = json.loads(mp.read_text(encoding="utf-8"))
            if m.get("status") == "COMPLETED" and m.get("training_time"):
                times.append(m["training_time"])
    except Exception:
        pass
    if len(times) < 3:
        return configured
    times.sort()
    median_min = times[len(times) // 2] / 60.0
    adaptive = max(5, int(median_min * 2 + 0.5))
    effective = min(configured, adaptive)
    if effective != configured:
        logger.debug("Adaptive stall: median=%.1fm → effective=%dm (configured=%dm)",
                     median_min, effective, configured)
    return effective


def check_active(active: Dict[int, TrainingProcess], results_dir: Path,
                 cfg: dict, failure_counts: dict,
                 fix_counts: Optional[dict] = None) -> list:
    """Check running processes. Reap completed/timed-out/stalled/OOM.
    Returns list of (idea_id, gpu) tuples for finished ideas.

    When fix_counts is provided and max_fix_attempts > 0, failed ideas
    are sent to the executor LLM for diagnosis before recording failure.
    If the LLM applies a fix, the idea is re-launched on the same GPU.
    """
    finished = []
    stall_minutes = _adaptive_stall_minutes(
        results_dir, cfg.get("stall_minutes", 0))
    if fix_counts is None:
        fix_counts = {}

    for gpu in list(active.keys()):
        tp = active[gpu]
        ret = tp.process.poll()
        elapsed = time.time() - tp.start_time

        # --- Still running ---
        if ret is None:
            if elapsed > tp.timeout:
                logger.warning("[TIMEOUT] %s after %.0fm — killing",
                               tp.idea_id, elapsed / 60)
                notify("stall", {"idea_id": tp.idea_id, "gpu": gpu,
                                 "reason": f"Timeout after {elapsed / 60:.0f}m"}, cfg)
                _terminate_and_reap(tp.process, tp.idea_id)
                tp.close_log()
                error_msg = "Timed out"
                if _try_executor_fix(tp.idea_id, error_msg,
                                     results_dir, cfg, fix_counts):
                    _reset_idea_for_retry(results_dir / tp.idea_id)
                    try:
                        new_tp = launch(tp.idea_id, gpu, results_dir, cfg)
                        active[gpu] = new_tp
                        logger.info("[FIX-RETRY] %s relaunched on GPU %d",
                                     tp.idea_id, gpu)
                        continue
                    except Exception as e:
                        logger.error("[FIX-RETRY] %s relaunch failed: %s",
                                      tp.idea_id, e)
                _write_failure(results_dir / tp.idea_id, error_msg)
                _record_failure(failure_counts, tp.idea_id)
                del active[gpu]
                finished.append((tp.idea_id, gpu))
                continue

            if check_stalled(tp, stall_minutes):
                logger.warning("[STALLED] %s — no log output for %dm, killing",
                               tp.idea_id, stall_minutes)
                notify("stall", {"idea_id": tp.idea_id, "gpu": gpu,
                                 "reason": f"Stalled ({stall_minutes}m no output)"}, cfg)
                _terminate_and_reap(tp.process, tp.idea_id)
                tp.close_log()
                error_msg = f"Stalled (no output for {stall_minutes}m)"
                if _try_executor_fix(tp.idea_id, error_msg,
                                     results_dir, cfg, fix_counts):
                    _reset_idea_for_retry(results_dir / tp.idea_id)
                    try:
                        new_tp = launch(tp.idea_id, gpu, results_dir, cfg)
                        active[gpu] = new_tp
                        logger.info("[FIX-RETRY] %s relaunched on GPU %d",
                                     tp.idea_id, gpu)
                        continue
                    except Exception as e:
                        logger.error("[FIX-RETRY] %s relaunch failed: %s",
                                      tp.idea_id, e)
                _write_failure(results_dir / tp.idea_id, error_msg)
                _record_failure(failure_counts, tp.idea_id)
                del active[gpu]
                finished.append((tp.idea_id, gpu))
                continue

            fatal = detect_fatal_in_log(tp)
            if fatal and tp.process.poll() is None:
                logger.warning("[FATAL-HUNG] %s — fatal error in log but "
                               "process still alive, killing:\n%s",
                               tp.idea_id, fatal[:200])
                notify("stall", {"idea_id": tp.idea_id, "gpu": gpu,
                                 "reason": f"Fatal error (hung): {fatal[:100]}"}, cfg)
                _terminate_and_reap(tp.process, tp.idea_id)
                tp.close_log()
                error_msg = f"Process hung after fatal error:\n{fatal[:500]}"
                if _try_executor_fix(tp.idea_id, error_msg,
                                     results_dir, cfg, fix_counts):
                    _reset_idea_for_retry(results_dir / tp.idea_id)
                    try:
                        new_tp = launch(tp.idea_id, gpu, results_dir, cfg)
                        active[gpu] = new_tp
                        logger.info("[FIX-RETRY] %s relaunched on GPU %d",
                                     tp.idea_id, gpu)
                        continue
                    except Exception as e:
                        logger.error("[FIX-RETRY] %s relaunch failed: %s",
                                      tp.idea_id, e)
                _write_failure(results_dir / tp.idea_id, error_msg)
                _record_failure(failure_counts, tp.idea_id)
                del active[gpu]
                finished.append((tp.idea_id, gpu))
                continue

            kill_file = results_dir / tp.idea_id / ".kill"
            if kill_file.exists():
                logger.info("Admin kill signal for %s — terminating", tp.idea_id)
                tp.process.terminate()
                kill_file.unlink(missing_ok=True)

            continue

        # --- Process exited ---
        # Reap zombie to prevent accumulation
        try:
            tp.process.wait(timeout=1)
        except Exception:
            pass
        tp.close_log()
        metrics_path = results_dir / tp.idea_id / "metrics.json"

        if ret == 0 and metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                metrics = {"status": "UNKNOWN"}
            status = metrics.get("status", "COMPLETED")
            logger.info("[%s] %s on GPU %d in %.1fm",
                        status, tp.idea_id, gpu, elapsed / 60)
            if status == "FAILED":
                error_msg = metrics.get("error", "Training script reported FAILED")
                if _try_executor_fix(tp.idea_id, error_msg,
                                     results_dir, cfg, fix_counts):
                    _reset_idea_for_retry(results_dir / tp.idea_id)
                    try:
                        new_tp = launch(tp.idea_id, gpu, results_dir, cfg)
                        active[gpu] = new_tp
                        logger.info("[FIX-RETRY] %s relaunched on GPU %d",
                                     tp.idea_id, gpu)
                        continue
                    except Exception as e:
                        logger.error("[FIX-RETRY] %s relaunch failed: %s",
                                      tp.idea_id, e)
                _record_failure(failure_counts, tp.idea_id)
        else:
            reason = f"exit code {ret}"
            try:
                tail_str = tail_file(tp.log_path, 8192)
                lines = tail_str.strip().split("\n")
                tail = "\n".join(lines[-5:])
                reason += f"\n{tail}"
            except Exception:
                pass
            logger.warning("[FAILED] %s on GPU %d — %s", tp.idea_id, gpu, reason)
            if _try_executor_fix(tp.idea_id, reason,
                                 results_dir, cfg, fix_counts):
                _reset_idea_for_retry(results_dir / tp.idea_id)
                try:
                    new_tp = launch(tp.idea_id, gpu, results_dir, cfg)
                    active[gpu] = new_tp
                    logger.info("[FIX-RETRY] %s relaunched on GPU %d",
                                 tp.idea_id, gpu)
                    continue
                except Exception as e:
                    logger.error("[FIX-RETRY] %s relaunch failed: %s",
                                  tp.idea_id, e)
            if not metrics_path.exists():
                _write_failure(results_dir / tp.idea_id,
                               f"Process exited with code {ret}")
            _record_failure(failure_counts, tp.idea_id)

        del active[gpu]
        finished.append((tp.idea_id, gpu))

    return finished


def _record_failure(failure_counts: dict, idea_id: str):
    """Increment failure count for an idea."""
    failure_counts[idea_id] = failure_counts.get(idea_id, 0) + 1


def get_skipped_ideas(failure_counts: dict, max_failures: int) -> set:
    """Return set of idea IDs that have exceeded max failure count."""
    if max_failures <= 0:
        return set()
    return {iid for iid, count in failure_counts.items()
            if count >= max_failures}


# ---------------------------------------------------------------------------
# Executor LLM fix — auto-diagnose and fix failed ideas
# ---------------------------------------------------------------------------

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

    train_script = cfg.get("train_script", "train_idea.py")

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


# ---------------------------------------------------------------------------
# Post-training evaluation
# ---------------------------------------------------------------------------

def launch_eval(idea_id: str, gpu: int, results_dir: Path,
                cfg: dict) -> Optional[EvalProcess]:
    """Launch a non-blocking eval subprocess. Returns EvalProcess or None."""
    eval_script = cfg.get("eval_script")
    if not eval_script:
        return None

    eval_output = cfg.get("eval_output") or "eval_report.json"
    output_path = results_dir / idea_id / eval_output
    if output_path.exists():
        logger.debug("Eval already exists for %s, skipping", idea_id)
        return None

    metrics_path = results_dir / idea_id / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics.get("status") != "COMPLETED":
                return None
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None
    else:
        return None

    python = cfg.get("python", sys.executable)
    eval_args = cfg.get("eval_args") or []
    eval_timeout = cfg.get("eval_timeout", 3600)

    cmd = [python, eval_script]
    cmd.extend(_format_args(eval_args, {"idea_id": idea_id, "gpu": gpu}))

    log_path = results_dir / idea_id / "eval_output.log"
    logger.info("Launching eval for %s on GPU %d", idea_id, gpu)

    try:
        env = os.environ.copy()
        for k, v in (cfg.get("train_extra_env") or {}).items():
            env[k] = str(v)
        # Don't set CUDA_VISIBLE_DEVICES — the eval script uses --gpu
        # to select the device via torch.cuda.set_device().
        env.pop("CUDA_VISIBLE_DEVICES", None)
        log_fh = open(log_path, "w", encoding="utf-8")
        proc = subprocess.Popen(
            cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
            preexec_fn=_new_process_group,
        )
        return EvalProcess(
            idea_id=idea_id, gpu=gpu, process=proc,
            start_time=time.time(), log_path=log_path,
            timeout=eval_timeout, _log_fh=log_fh,
        )
    except Exception as e:
        logger.warning("Failed to launch eval for %s: %s", idea_id, e)
        return None


def run_eval(idea_id: str, gpu: int, results_dir: Path, cfg: dict):
    """Run post-training evaluation (blocking). Used in --once mode."""
    ep = launch_eval(idea_id, gpu, results_dir, cfg)
    if ep is None:
        return
    eval_output = cfg.get("eval_output") or "eval_report.json"
    reason = ""
    try:
        ep.process.wait(timeout=ep.timeout)
        if ep.process.returncode == 0:
            logger.info("Eval completed for %s", idea_id)
        else:
            reason = f"Exit code {ep.process.returncode}"
            logger.warning("Eval failed for %s (exit %d)",
                           idea_id, ep.process.returncode)
    except subprocess.TimeoutExpired:
        reason = f"Timed out after {ep.timeout}s"
        logger.warning("Eval timed out for %s after %ds",
                       idea_id, ep.timeout)
        _terminate_and_reap(ep.process, f"eval {idea_id}")
    except Exception as e:
        reason = str(e)
        logger.warning("Eval error for %s: %s", idea_id, e)
    finally:
        ep.close_log()
        if reason:
            _write_eval_failure_marker(results_dir, idea_id, eval_output, reason)


def _write_eval_failure_marker(results_dir: Path, idea_id: str,
                               eval_output: str, reason: str) -> None:
    """Safety net: write failure marker if eval process died without one.

    The marker file is the eval_output itself so the backlog scanner
    won't re-queue this idea.  The eval script is responsible for
    writing domain-specific reports; this is a generic fallback.
    """
    report_path = results_dir / idea_id / eval_output
    if report_path.exists():
        return  # Script already wrote a report
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({
        "status": "FAILED",
        "reason": reason[:500],
    }, indent=2))
    logger.info("Wrote eval failure marker for %s", idea_id)


def check_active_evals(active_evals: Dict[int, EvalProcess],
                       results_dir: Path, cfg: dict) -> list:
    """Check running eval processes. Returns list of (idea_id, gpu) for finished evals."""
    eval_output = cfg.get("eval_output") or "eval_report.json"
    finished = []
    for gpu in list(active_evals.keys()):
        ep = active_evals[gpu]
        ret = ep.process.poll()
        elapsed = time.time() - ep.start_time

        if ret is None:
            # Still running — check timeout
            if elapsed > ep.timeout:
                logger.warning("[EVAL TIMEOUT] %s after %.0fm — killing",
                               ep.idea_id, elapsed / 60)
                _terminate_and_reap(ep.process, f"eval {ep.idea_id}")
                ep.close_log()
                _write_eval_failure_marker(
                    results_dir, ep.idea_id, eval_output,
                    f"Timed out after {elapsed/60:.0f}m")
                del active_evals[gpu]
                finished.append((ep.idea_id, gpu))
            continue

        # Process exited
        ep.close_log()
        if ret == 0:
            logger.info("[EVAL OK] %s on GPU %d in %.1fm",
                        ep.idea_id, gpu, elapsed / 60)
        else:
            # Log tail of eval output for diagnosis
            eval_tail = tail_file(ep.log_path, 2048).strip()
            logger.warning("[EVAL FAILED] %s on GPU %d — exit %d\n%s",
                           ep.idea_id, gpu, ret,
                           eval_tail[-500:] if eval_tail else "(no output)")
            _write_eval_failure_marker(
                results_dir, ep.idea_id, eval_output,
                f"Exit code {ret}: {eval_tail[-300:] if eval_tail else 'no output'}")
        del active_evals[gpu]
        finished.append((ep.idea_id, gpu))

    return finished


def check_active_roles(active_roles: Dict[str, "RoleProcess"],
                       ideas_file: str = "ideas.md") -> list:
    """Check running role processes. Returns list of (role_name, success) tuples."""
    finished = []
    for role_name in list(active_roles.keys()):
        rp = active_roles[role_name]
        ret = rp.process.poll()
        elapsed = time.time() - rp.start_time

        if ret is None:
            # Still running — check timeout
            if elapsed > rp.timeout:
                logger.warning("[ROLE TIMEOUT] %s after %.0fm — killing",
                               role_name, elapsed / 60)
                _terminate_and_reap(rp.process, f"role {role_name}")
                rp.close_log()
                _fs_unlock(rp.lock_dir)
                del active_roles[role_name]
                finished.append((role_name, False))
            continue

        # Process exited
        rp.close_log()
        _fs_unlock(rp.lock_dir)
        if ret == 0:
            logger.info("%s cycle %d completed", role_name, rp.cycle_num)
        else:
            logger.warning("%s cycle %d failed (exit %d), see %s",
                           role_name, rp.cycle_num, ret, rp.log_path)

        # CORRUPTION GUARD: check if ideas.md was truncated by the role
        _check_ideas_integrity(ideas_file, rp)

        del active_roles[role_name]
        finished.append((role_name, ret == 0))

    return finished


def _check_ideas_integrity(ideas_file: str, rp: "RoleProcess"):
    """Detect and auto-restore ideas.md if a role truncated/corrupted it.

    Compares current file size and idea count against pre-role snapshot.
    If file shrunk by >10% or lost ideas, restore from the .safe backup.
    """
    ideas_path = Path(ideas_file)
    if not ideas_path.exists() or rp.ideas_pre_size == 0:
        return

    current_size = ideas_path.stat().st_size
    # Quick check: if file grew or stayed same, it's fine
    if current_size >= rp.ideas_pre_size:
        return

    # File shrunk — check idea count to confirm corruption
    try:
        current_text = ideas_path.read_text(encoding="utf-8")
        current_count = len(re.findall(r"^## idea-\d+:", current_text,
                                       re.MULTILINE))
    except OSError:
        current_count = 0

    if current_count >= rp.ideas_pre_count:
        return  # Count OK despite smaller size (unlikely but possible)

    # CORRUPTION DETECTED
    shrink_pct = (1 - current_size / rp.ideas_pre_size) * 100
    lost_ideas = rp.ideas_pre_count - current_count
    logger.error(
        "IDEAS.MD CORRUPTION DETECTED after %s cycle %d! "
        "File shrunk %.0f%% (%d→%d bytes), lost %d ideas (%d→%d). "
        "Restoring from backup.",
        rp.role_name, rp.cycle_num, shrink_pct,
        rp.ideas_pre_size, current_size,
        lost_ideas, rp.ideas_pre_count, current_count)

    # Restore from .safe backup — validate by idea count, not byte size
    backup_path = ideas_path.with_suffix(".md.safe")
    if backup_path.exists():
        try:
            backup_text = backup_path.read_text(encoding="utf-8")
            backup_count = len(re.findall(r"^## idea-\d+:", backup_text,
                                          re.MULTILINE))
        except OSError:
            backup_count = 0

        if backup_count >= rp.ideas_pre_count:
            # Save the corrupted file for forensics
            corrupt_path = ideas_path.with_suffix(
                f".md.corrupt.{int(time.time())}")
            import shutil
            shutil.copy2(str(ideas_path), str(corrupt_path))
            # Restore
            shutil.copy2(str(backup_path), str(ideas_path))
            logger.info("Restored ideas.md from backup (%d ideas). "
                        "Corrupted version saved to %s",
                        backup_count, corrupt_path)
        else:
            logger.error("Backup also has fewer ideas (%d vs expected %d). "
                         "NOT restoring.",
                         backup_count, rp.ideas_pre_count)
    else:
        logger.error("No backup found at %s — cannot restore!", backup_path)


def run_post_scripts(idea_id: str, gpu: int, results_dir: Path, cfg: dict):
    """Run additional post-training scripts (beyond eval_script).
    Each entry in post_scripts is a dict with: script, args, timeout, output."""
    post_scripts = (cfg.get("post_scripts") or [])
    if not post_scripts:
        return

    # Check training succeeded
    metrics_path = results_dir / idea_id / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            if metrics.get("status") != "COMPLETED":
                return
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return
    else:
        return

    python = cfg.get("python", sys.executable)
    env = os.environ.copy()
    for k, v in (cfg.get("train_extra_env") or {}).items():
        env[k] = str(v)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)

    for i, ps in enumerate(post_scripts):
        script = ps.get("script")
        if not script:
            continue

        # Skip if output already exists
        output_file = ps.get("output", "")
        if output_file:
            output_path = results_dir / idea_id / output_file
            if output_path.exists():
                logger.debug("Post-script %d output exists for %s, skipping",
                             i, idea_id)
                continue

        args = ps.get("args") or []
        timeout = ps.get("timeout", 3600)
        name = ps.get("name", f"post-script-{i}")

        cmd = [python, script]
        cmd.extend(_format_args(args, {"idea_id": idea_id, "gpu": gpu}))

        log_path = results_dir / idea_id / f"{name}.log"
        logger.info("Running %s for %s", name, idea_id)

        try:
            with open(log_path, "w", encoding="utf-8") as log_fh:
                result = subprocess.run(
                    cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
                    timeout=timeout,
                )
            if result.returncode == 0:
                logger.info("%s completed for %s", name, idea_id)
            else:
                logger.warning("%s failed for %s (exit %d)",
                               name, idea_id, result.returncode)
        except subprocess.TimeoutExpired:
            logger.warning("%s timed out for %s after %ds",
                           name, idea_id, timeout)
        except Exception as e:
            logger.warning("%s error for %s: %s", name, idea_id, e)


# ---------------------------------------------------------------------------
# Garbage collection / cleanup
# ---------------------------------------------------------------------------

def run_cleanup(results_dir: Path, cfg: dict):
    """Run periodic cleanup: GC checkpoints, delete file patterns, run script."""
    cleanup_cfg = cfg.get("cleanup") or {}

    # GC: delete checkpoint dirs for non-top experiments
    gc_cfg = cfg.get("gc") or {}
    if gc_cfg.get("enabled") and gc_cfg.get("checkpoints_dir"):
        try:
            from orze_gc import run_gc  # orze/gc.py (aliased to avoid stdlib gc)
            report_cfg = cfg.get("report") or {}
            ideas_path = Path(cfg.get("ideas_file", "ideas.md"))
            lake_path = ideas_path.parent / "idea_lake.db"
            stats = run_gc(
                results_dir=results_dir,
                checkpoints_dir=Path(gc_cfg["checkpoints_dir"]),
                primary_metric=report_cfg.get("primary_metric", ""),
                lake_db_path=lake_path if lake_path.exists() else None,
                keep_top=gc_cfg.get("keep_top", 50),
                keep_recent=gc_cfg.get("keep_recent", 20),
                min_free_gb=gc_cfg.get("min_free_gb", 0),
            )
            cs = stats.get("checkpoints", {})
            if cs.get("deleted", 0) > 0:
                logger.info("GC: deleted %d checkpoint dirs, kept %d",
                            cs["deleted"], cs["kept"])
        except Exception as e:
            logger.warning("GC failed: %s", e)

    # Built-in: delete files matching glob patterns in results dirs
    patterns = cleanup_cfg.get("patterns") or []
    if patterns:
        deleted = 0
        for d in results_dir.iterdir():
            if not d.is_dir() or not d.name.startswith("idea-"):
                continue
            for pattern in patterns:
                for f in d.glob(pattern):
                    try:
                        if f.is_file():
                            f.unlink()
                            deleted += 1
                    except Exception:
                        pass
        if deleted:
            logger.info("Cleanup: deleted %d files matching %s",
                        deleted, patterns)

    # Custom cleanup script
    script = cleanup_cfg.get("script")
    if script:
        python = cfg.get("python", sys.executable)
        timeout = cleanup_cfg.get("timeout", 300)
        try:
            result = subprocess.run(
                [python, script], capture_output=True, text=True,
                timeout=timeout,
            )
            if result.returncode == 0:
                logger.info("Cleanup script completed")
            else:
                logger.warning("Cleanup script failed (exit %d)",
                               result.returncode)
        except Exception as e:
            logger.warning("Cleanup script error: %s", e)


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------

def _read_metric_value(results_dir: Path, idea_id: str, col: dict) -> Any:
    """Read a metric value for a column. Supports 'source' field for
    reading from other JSON files: 'filename.json:dotpath'."""
    source = col.get("source", "")
    key = col.get("key")
    if not key:
        return None

    if source and ":" in source:
        filename, dotpath = source.split(":", 1)
        filepath = results_dir / idea_id / filename
        if filepath.exists():
            try:
                data = json.loads(filepath.read_text(encoding="utf-8"))
                return deep_get(data, dotpath)
            except (json.JSONDecodeError, KeyError, OSError, UnicodeDecodeError):
                return None
        return None

    metrics_path = results_dir / idea_id / "metrics.json"
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text(encoding="utf-8"))
            return deep_get(data, key) if "." in key else data.get(key)
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            return None
    return None


def update_report(results_dir: Path, ideas: Dict[str, dict],
                  cfg: dict) -> list:
    """Generate a configurable leaderboard report.md from all results.
    Returns sorted list of completed row dicts."""
    report_cfg = cfg.get("report") or DEFAULT_CONFIG["report"]
    primary_metric = report_cfg.get("primary_metric") or "test_accuracy"
    sort_order = report_cfg.get("sort") or "descending"
    columns = report_cfg.get("columns") or DEFAULT_CONFIG["report"]["columns"]
    title = report_cfg.get("title") or "Orze Report"

    rows = []
    def _id_sort_key(x):
        match = re.search(r"\d+", x)
        return int(match.group()) if match else 999999

    # Include archived ideas from cached index
    all_ideas = dict(ideas)
    archived_index = results_dir / "_archived_index.json"
    if archived_index.exists():
        try:
            idx = json.loads(archived_index.read_text(encoding="utf-8"))
            for arch_id, arch_title in idx.items():
                if arch_id not in all_ideas:
                    all_ideas[arch_id] = {"title": arch_title, "priority": "archived"}
        except (json.JSONDecodeError, OSError):
            pass

    for idea_id in sorted(all_ideas.keys(), key=_id_sort_key):
        idea_dir = results_dir / idea_id

        if not idea_dir.exists():
            rows.append({"id": idea_id, "title": all_ideas[idea_id]["title"],
                         "status": "QUEUED", "values": {}})
            continue

        metrics_path = idea_dir / "metrics.json"
        if not metrics_path.exists():
            rows.append({"id": idea_id, "title": all_ideas[idea_id]["title"],
                         "status": "IN_PROGRESS", "values": {}})
            continue

        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            metrics = {"status": "FAILED", "error": "corrupt metrics.json"}

        values = {}
        for col in columns:
            key = col.get("key")
            if not key:
                continue
            values[key] = _read_metric_value(results_dir, idea_id, col)

        primary_val = values.get(primary_metric)
        if primary_val is None:
            primary_val = (deep_get(metrics, primary_metric)
                           if "." in primary_metric
                           else metrics.get(primary_metric))

        rows.append({
            "id": idea_id, "title": all_ideas[idea_id]["title"],
            "status": metrics.get("status", "UNKNOWN"),
            "values": values,
            "primary_val": primary_val,
            "metrics": metrics,
        })

    counts = {}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1

    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    lines = [
        f"# {title}",
        f"**Updated:** {now} | **Host:** {socket.gethostname()}",
        "",
        "## Pipeline Status",
        "| Total | Completed | Failed | In Progress | Queued |",
        "|-------|-----------|--------|-------------|--------|",
        f"| {len(rows)} | {counts.get('COMPLETED', 0)} "
        f"| {counts.get('FAILED', 0)} "
        f"| {counts.get('IN_PROGRESS', 0)} | {counts.get('QUEUED', 0)} |",
        "",
        "## Results",
        "",
    ]

    completed = [r for r in rows if r["status"] == "COMPLETED"]
    reverse = sort_order == "descending"
    _sentinel = float("-inf") if reverse else float("inf")

    def _safe_float(v):
        try:
            return float(v)
        except (ValueError, TypeError):
            return _sentinel

    completed.sort(key=lambda r: _safe_float(r.get("primary_val")),
                   reverse=reverse)

    # --- Sweep grouping: split into standalone + sweep groups ---
    standalone = []
    sweep_groups = {}  # parent_id -> list of sub-run rows
    for r in completed:
        if "~" in r["id"]:
            parent_id = r["id"].split("~", 1)[0]
            sweep_groups.setdefault(parent_id, []).append(r)
        else:
            standalone.append(r)

    # Build main table: standalone + best of each sweep group
    main_rows = list(standalone)
    for parent_id, children in sweep_groups.items():
        children.sort(key=lambda r: _safe_float(r.get("primary_val")),
                      reverse=reverse)
        best = dict(children[0])
        best["title"] = f"{best['title']} (best of {len(children)})"
        main_rows.append(best)

    main_rows.sort(key=lambda r: _safe_float(r.get("primary_val")),
                   reverse=reverse)

    if main_rows:
        header = "| Rank | Idea | Title"
        sep = "|------|------|------"
        for col in columns:
            label = col.get("label", col.get("key", "?"))
            header += f" | {label}"
            sep += " |" + "-" * max(6, len(str(label)))
        header += " |"
        sep += " |"
        lines.append(header)
        lines.append(sep)

        for rank, r in enumerate(main_rows, 1):
            row = f"| {rank} | {r['id']} | {r['title'][:50]}"
            for col in columns:
                key = col.get("key")
                if not key:
                    row += " | —"
                    continue
                val = r["values"].get(key)
                if val is not None:
                    fmt = col.get("fmt", "")
                    try:
                        row += f" | {val:{fmt}}"
                    except (ValueError, TypeError):
                        row += f" | {val}"
                else:
                    row += " | —"
            row += " |"
            lines.append(row)
        lines.append("")

    # --- Sweep Details section ---
    if sweep_groups:
        lines.append("## Sweep Details")
        lines.append("")
        for parent_id in sorted(sweep_groups.keys(), key=_id_sort_key):
            children = sweep_groups[parent_id]
            children.sort(key=lambda r: _safe_float(r.get("primary_val")),
                          reverse=reverse)
            lines.append(f"### {parent_id} ({len(children)} variants)")
            lines.append(f"| Rank | Sub-run | {primary_metric} |")
            lines.append("|------|---------|" + "-" * max(6, len(primary_metric)) + "|")
            for i, r in enumerate(children, 1):
                pv = r.get("primary_val", "—")
                if isinstance(pv, float):
                    pv = f"{pv:.4f}"
                suffix = r["id"].split("~", 1)[1] if "~" in r["id"] else r["id"]
                lines.append(f"| {i} | {suffix} | {pv} |")
            lines.append("")

    failed = [r for r in rows if r["status"] == "FAILED"]
    if failed:
        lines.append("## Failed")
        for r in failed:
            err = r.get("metrics", {}).get("error", "unknown")
            lines.append(
                f"- **{r['id']}**: {r['title'][:50]} — {str(err)[:80]}")
        lines.append("")

    queued = [r for r in rows if r["status"] == "QUEUED"]
    if queued:
        lines.append(f"## Queue ({len(queued)} ideas)")
        for r in queued[:20]:
            pri = all_ideas.get(r["id"], {}).get("priority", "medium")
            lines.append(f"- **{r['id']}** [{pri}]: {r['title'][:60]}")
        if len(queued) > 20:
            lines.append(f"- ... and {len(queued) - 20} more")
        lines.append("")

    report_path = results_dir / "report.md"
    atomic_write(report_path, "\n".join(lines))

    # Write leaderboard cache for admin panel (avoids expensive rescan)
    lb_entries = []
    for r in main_rows[:20]:
        lb_entries.append({
            "idea_id": r["id"],
            "title": r["title"],
            "metric_value": r.get("primary_val"),
            "training_time": r["values"].get("training_time"),
            "status": "COMPLETED",
            "eval_metrics": r["values"],
        })
    lb_path = results_dir / "_leaderboard.json"
    atomic_write(lb_path, json.dumps({"top": lb_entries, "metric": primary_metric},
                                     default=str))

    logger.info("Report updated: %d completed, %d queued, %d failed",
                counts.get("COMPLETED", 0), counts.get("QUEUED", 0),
                counts.get("FAILED", 0))

    return completed


def write_admin_cache(results_dir: Path, ideas: dict, cfg: dict):
    """Write pre-aggregated _admin_cache.json for instant admin panel access.

    Aggregates fleet (heartbeats + GPU info), queue (with status),
    and alerts — so the admin server never needs to scan the filesystem.
    """
    now = time.time()

    # --- Fleet: enrich heartbeats ---
    raw_hb = _read_all_heartbeats(results_dir, stale_seconds=600)
    heartbeats = []
    for hb in raw_hb:
        age = now - hb.get("epoch", 0)
        status = "online"
        if age > 300:
            status = "offline"
        elif age > 120:
            status = "degraded"
        heartbeats.append({
            **hb,
            "status": status,
            "heartbeat_age_sec": round(age, 1),
        })

    # --- Queue: expanded ideas with status ---
    sweep_max = cfg.get("sweep", {}).get("max_combos", 20)
    expanded = expand_sweeps(dict(ideas), max_combos=sweep_max)
    all_statuses: dict = {}
    queue_items = []
    for idea_id, idea in expanded.items():
        idea_dir = results_dir / idea_id
        idea_status = "pending"
        if idea_dir.exists():
            mpath = idea_dir / "metrics.json"
            if mpath.exists():
                try:
                    m = json.loads(mpath.read_text(encoding="utf-8"))
                    idea_status = m.get("status", "COMPLETED").lower()
                except (json.JSONDecodeError, OSError):
                    idea_status = "running"
            else:
                idea_status = "running"
        all_statuses[idea_status] = all_statuses.get(idea_status, 0) + 1
        queue_items.append({
            "idea_id": idea_id,
            "title": idea.get("title", ""),
            "priority": idea.get("priority", "medium"),
            "status": idea_status,
            "config": idea.get("config", {}),
            "sweep_parent": idea.get("_sweep_parent"),
        })

    # --- Alerts ---
    alerts = []
    two_hours_ago = now - 7200
    try:
        with os.scandir(results_dir) as it:
            for entry in it:
                if not entry.is_dir() or not entry.name.startswith("idea-"):
                    continue
                try:
                    mtime = entry.stat().st_mtime
                except OSError:
                    continue
                if mtime < two_hours_ago:
                    continue
                mpath = Path(entry.path) / "metrics.json"
                if not mpath.exists():
                    continue
                try:
                    m = json.loads(mpath.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    continue
                if m.get("status") not in ("FAILED", "ERROR"):
                    continue
                alerts.append({
                    "type": "failure",
                    "idea_id": entry.name,
                    "error": str(m.get("error", m.get("status", "")))[:200],
                    "minutes_ago": round((now - mtime) / 60, 1),
                })
    except OSError:
        pass

    for hb in heartbeats:
        if hb.get("status") == "offline":
            alerts.append({
                "type": "stale_host",
                "host": hb.get("host", "unknown"),
                "minutes_ago": round(hb.get("heartbeat_age_sec", 0) / 60, 1),
            })

    try:
        usage = shutil.disk_usage(results_dir)
        disk_free = round(usage.free / (1024 ** 3), 1)
        if disk_free < 50:
            alerts.append({"type": "low_disk", "disk_free_gb": disk_free})
    except Exception:
        pass

    cache = {
        "fleet": {"heartbeats": heartbeats, "local_gpus": []},
        "queue": {"items": queue_items, "counts": all_statuses,
                  "total_all": sum(all_statuses.values())},
        "alerts": {"alerts": alerts, "count": len(alerts)},
        "epoch": now,
    }
    atomic_write(results_dir / "_admin_cache.json",
                 json.dumps(cache, default=str))


# ---------------------------------------------------------------------------
# Status JSON (for LLM consumption)
# ---------------------------------------------------------------------------

def _query_gpu_details() -> List[dict]:
    """Query nvidia-smi for per-GPU stats (memory, util, temp)."""
    try:
        result = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        gpus = []
        for line in result.stdout.strip().splitlines():
            parts = [p.strip() for p in line.split(",")]
            if len(parts) >= 6:
                gpus.append({
                    "index": int(parts[0]),
                    "name": parts[1],
                    "memory_used_mib": int(parts[2]),
                    "memory_total_mib": int(parts[3]),
                    "utilization_pct": int(parts[4]),
                    "temperature_c": int(parts[5]),
                })
        return gpus
    except Exception:
        return []


def write_host_heartbeat(results_dir: Path,
                         active: Dict[int, TrainingProcess],
                         free_gpus: List[int]):
    """Write per-host+PID heartbeat file with active processes and free GPUs.
    Other hosts read these to build a merged multi-machine view."""
    hostname = socket.gethostname()
    pid = os.getpid()
    now = time.time()
    heartbeat = {
        "host": hostname,
        "pid": pid,
        "timestamp": datetime.datetime.now().isoformat(),
        "epoch": now,
        "active": [
            {
                "idea_id": tp.idea_id,
                "gpu": tp.gpu,
                "elapsed_min": round((now - tp.start_time) / 60, 1),
            }
            for tp in active.values()
        ],
        "free_gpus": free_gpus,
        "gpu_info": _query_gpu_details(),
        "os": f"{platform.system()} {platform.release()}",
    }
    atomic_write(results_dir / f"_host_{hostname}_{pid}.json",
                 json.dumps(heartbeat, indent=2))


def _read_all_heartbeats(results_dir: Path,
                         stale_seconds: float = 300) -> list:
    """Read all host heartbeat files, filtering out stale ones (>5min).
    Removes stale heartbeat files to prevent clutter."""
    now = time.time()
    heartbeats = []
    for hb_path in results_dir.glob("_host_*.json"):
        try:
            hb = json.loads(hb_path.read_text(encoding="utf-8"))
            age = now - hb.get("epoch", 0)
            if age <= stale_seconds:
                heartbeats.append(hb)
            else:
                # Clean up stale heartbeat file
                try:
                    hb_path.unlink()
                except OSError:
                    pass
        except Exception:
            # Purge unparseable/corrupt heartbeat files if stale
            try:
                if now - hb_path.stat().st_mtime > stale_seconds:
                    hb_path.unlink()
            except OSError:
                pass
            continue
    return heartbeats


def write_status_json(results_dir: Path, iteration: int,
                      active: Dict[int, TrainingProcess],
                      free_gpus: List[int], queue_depth: int,
                      completed_count: int, failed_count: int,
                      skipped_count: int, top_results: list,
                      cfg: dict,
                      role_states: Optional[dict] = None):
    """Write machine-readable status.json for LLM agents.
    Merges heartbeats from all hosts for a combined multi-machine view."""
    disk_free_gb = 0.0
    try:
        usage = shutil.disk_usage(results_dir)
        disk_free_gb = usage.free / (1024 ** 3)
    except Exception:
        pass

    now = time.time()

    # Merge active processes from all hosts
    heartbeats = _read_all_heartbeats(results_dir)
    all_active = []
    free_gpus_by_host = {}
    for hb in heartbeats:
        host = hb.get("host", "unknown")
        for a in hb.get("active", []):
            a["host"] = host
            all_active.append(a)
        free_gpus_by_host[host] = hb.get("free_gpus", [])

    # Build per-role status
    role_states = role_states or {}
    roles_cfg = cfg.get("roles") or {}
    roles_status = {}
    for rname in roles_cfg:
        rs = role_states.get(rname, {})
        last_run = rs.get("last_run_time", 0.0)
        roles_status[rname] = {
            "enabled": True,
            "cycles": rs.get("cycles", 0),
            "last_run_min_ago": (
                round((now - last_run) / 60, 1) if last_run > 0 else None
            ),
        }

    # Backward compat: flat research_* keys
    research_rs = role_states.get("research", {})
    research_last = research_rs.get("last_run_time", 0.0)

    hostname = socket.gethostname()
    status = {
        "timestamp": datetime.datetime.now().isoformat(),
        "host": hostname,
        "iteration": iteration,
        "active": all_active,
        "free_gpus": free_gpus,
        "free_gpus_by_host": free_gpus_by_host,
        "queue_depth": queue_depth,
        "completed": completed_count,
        "failed": failed_count,
        "skipped": skipped_count,
        "disk_free_gb": round(disk_free_gb, 1),
        "top_results": top_results[:10],
        "roles": roles_status,
        "research_enabled": "research" in roles_cfg,
        "research_cycles": research_rs.get("cycles", 0),
        "last_research_min_ago": (
            round((now - research_last) / 60, 1) if research_last > 0
            else None
        ),
    }

    atomic_write(results_dir / "status.json", json.dumps(status, indent=2))


# ---------------------------------------------------------------------------
# Notifications
# ---------------------------------------------------------------------------

def _notify_send(url: str, payload: dict,
                 headers: Optional[Dict[str, str]] = None,
                 timeout: int = 10):
    """POST JSON payload to a URL. Never raises."""
    try:
        data = json.dumps(payload).encode("utf-8")
        req_headers = {
            "Content-Type": "application/json",
            "User-Agent": f"orze/{__version__}",
        }
        if headers:
            req_headers.update(headers)
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers=req_headers)
        urllib.request.urlopen(req, timeout=timeout)
    except Exception as e:
        logger.warning("Notification failed (%s): %s", url[:60], e)


def _format_leaderboard(data: dict, bold_fn=str, escape_fn=str) -> str:
    """Format top 10 leaderboard lines from data['leaderboard'].
    escape_fn is applied to all text content (needed for Telegram HTML)."""
    board = data.get("leaderboard", [])
    if not board:
        return ""
    metric = escape_fn(str(data.get("metric_name", "score")))
    lines = [f"\nTop {len(board)} ({metric}):"]
    for i, entry in enumerate(board, 1):
        val = entry.get("value")
        val_str = escape_fn(f"{val:.4f}" if isinstance(val, float) else str(val))
        marker = escape_fn(" <-" if entry["id"] == data.get("idea_id") else "")
        title = escape_fn(str(entry.get("title", ""))[:30])
        line = f"#{i} {escape_fn(str(entry['id']))}: {val_str} {title}{marker}"
        if i == 1:
            line = bold_fn(line)
        lines.append(line)
    return "\n".join(lines)


def _format_report_text(data: dict, monospace: bool = False) -> str:
    """Format a periodic report summary.

    data keys:
      title, completed, failed, active_count, queued,
      leaderboard: [{id, title, value}],
      metric_name,
      machines: [{host, gpus_busy, gpus_total, utilization}]
    """
    c = data.get("completed", 0)
    f = data.get("failed", 0)
    a = data.get("active_count", 0)
    q = data.get("queued", 0)
    title = data.get("title", "Report")
    metric = data.get("metric_name", "score")
    board = data.get("leaderboard", [])
    machines = data.get("machines", [])

    lines = [title]
    lines.append(f"{c} completed | {f} failed | {a} active | {q} queued")
    lines.append("")

    # Machine status
    if machines:
        lines.append("Machines:")
        for m in machines:
            host = m.get("host", "?")
            busy = m.get("gpus_busy", 0)
            total = m.get("gpus_total", 0)
            util = m.get("utilization", "?")
            lines.append(f"  {host}: {busy}/{total} GPUs, {util}% util")
        lines.append("")

    # Top 10 leaderboard
    if board:
        lines.append(f"Top {len(board)} ({metric}):")
        for i, entry in enumerate(board, 1):
            val = entry.get("value")
            val_str = f"{val:.4f}" if isinstance(val, float) else str(val)
            eid = entry.get("id", "?")
            title_short = entry.get("title", "")[:25]
            lines.append(f"  #{i} {eid}: {val_str} {title_short}")

    return "\n".join(lines)


def _format_slack(event: str, data: dict) -> dict:
    """Format notification for Slack webhook."""
    if event == "report":
        return {"text": f"```\n{_format_report_text(data)}\n```"}
    if event in ("started", "shutdown"):
        icon = ":arrow_forward:" if event == "started" else ":stop_button:"
        host = data.get("host", socket.gethostname())
        return {"text": f"{icon} *Orze {event}* on `{host}`\n{data.get('message', '')}"}
    if event == "heartbeat":
        host = data.get("host", socket.gethostname())
        return {"text": (f":green_heart: *Heartbeat* on `{host}` | "
                         f"Iter {data.get('iteration', '?')} | "
                         f"Up {data.get('uptime', '?')} | "
                         f"{data.get('training', 0)}T/{data.get('eval', 0)}E/"
                         f"{data.get('free', 0)}F GPUs | "
                         f"Done {data.get('completed', 0)} | "
                         f"Q {data.get('queued', 0)}")}
    if event == "milestone":
        return {"text": f":dart: *Milestone: {data.get('count', '?')} experiments completed!*"}
    if event == "disk_warning":
        return {"text": f":warning: *Low disk* on `{data.get('host', socket.gethostname())}` — "
                        f"only {data.get('free_gb', '?')}GB free"}
    if event == "stall":
        return {"text": f":rotating_light: *{data.get('reason', 'Stalled')}*: "
                        f"`{data.get('idea_id', '?')}` on GPU {data.get('gpu', '?')}"}
    if event == "role_summary":
        return {"text": f":test_tube: *{data.get('role', '?')}* finished | "
                        f"{data.get('new_ideas', 0)} new ideas | {data.get('queued', '?')} queued"}
    if event == "new_best":
        prev = data.get("prev_best_id", "none")
        text = (f":trophy: *NEW BEST* `{data['idea_id']}`: {data['title']}\n"
                f"{data['metric_name']}: *{data['metric_value']}*"
                f" (was `{prev}`)")
    elif event == "failed":
        err = str(data.get("error") or "unknown")[:200]
        text = (f":x: *FAILED* `{data['idea_id']}`: {data['title']}\n"
                f"Error: {err}")
    else:
        t_str = ""
        try:
            t = data.get("training_time")
            if t is not None:
                t_str = f" in {float(t):.0f}s"
        except (ValueError, TypeError):
            pass
        text = (f":white_check_mark: *Completed* `{data['idea_id']}`: "
                f"{data['title']}\n"
                f"{data['metric_name']}: {data['metric_value']}"
                f" (rank #{data.get('rank', '?')})"
                f"{t_str}")
    text += _format_leaderboard(data, lambda s: f"*{s}*")
    return {"text": text}


def _format_discord(event: str, data: dict) -> dict:
    """Format notification for Discord webhook."""
    if event == "report":
        return {"content": f"```\n{_format_report_text(data)}\n```"}
    if event in ("started", "shutdown"):
        icon = "\u25b6\ufe0f" if event == "started" else "\u23f9\ufe0f"
        host = data.get("host", socket.gethostname())
        return {"content": f"{icon} **Orze {event}** on `{host}`\n{data.get('message', '')}"}
    if event == "heartbeat":
        host = data.get("host", socket.gethostname())
        return {"content": (f"\U0001f49a **Heartbeat** on `{host}` | "
                            f"Iter {data.get('iteration', '?')} | "
                            f"Up {data.get('uptime', '?')} | "
                            f"{data.get('training', 0)}T/{data.get('eval', 0)}E/"
                            f"{data.get('free', 0)}F GPUs | "
                            f"Done {data.get('completed', 0)} | Q {data.get('queued', 0)}")}
    if event == "milestone":
        return {"content": f"\U0001f3af **Milestone: {data.get('count', '?')} experiments completed!**"}
    if event == "disk_warning":
        return {"content": f"\u26a0\ufe0f **Low disk** on `{data.get('host', socket.gethostname())}` — "
                           f"only {data.get('free_gb', '?')}GB free"}
    if event == "stall":
        return {"content": f"\U0001f6a8 **{data.get('reason', 'Stalled')}**: "
                           f"`{data.get('idea_id', '?')}` on GPU {data.get('gpu', '?')}"}
    if event == "role_summary":
        return {"content": f"\U0001f9ea **{data.get('role', '?')}** finished | "
                           f"{data.get('new_ideas', 0)} new ideas | {data.get('queued', '?')} queued"}
    if event == "new_best":
        prev = data.get("prev_best_id", "none")
        content = (f"**NEW BEST** `{data['idea_id']}`: {data['title']}\n"
                   f"{data['metric_name']}: **{data['metric_value']}**"
                   f" (was `{prev}`)")
    elif event == "failed":
        err = str(data.get("error") or "unknown")[:200]
        content = (f"**FAILED** `{data['idea_id']}`: {data['title']}\n"
                   f"Error: {err}")
    else:
        t_str = ""
        try:
            t = data.get("training_time")
            if t is not None:
                t_str = f" in {float(t):.0f}s"
        except (ValueError, TypeError):
            pass
        content = (f"**Completed** `{data['idea_id']}`: {data['title']}\n"
                   f"{data['metric_name']}: {data['metric_value']}"
                   f" (rank #{data.get('rank', '?')})"
                   f"{t_str}")
    content += _format_leaderboard(data, lambda s: f"**{s}**")
    return {"content": content}


def _format_telegram(event: str, data: dict, channel_cfg: dict) -> tuple:
    """Format notification for Telegram Bot API using HTML parse_mode.
    Returns (url, payload). Uses HTML to avoid MarkdownV2 escaping issues."""
    esc = html_mod.escape
    token = channel_cfg["bot_token"]
    chat_id = channel_cfg["chat_id"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    if event == "report":
        text = f"<pre>\n{esc(_format_report_text(data))}\n</pre>"
        return url, {"chat_id": chat_id, "text": text,
                     "parse_mode": "HTML"}

    if event in ("started", "shutdown"):
        host = esc(data.get("host", socket.gethostname()))
        msg = esc(str(data.get("message", "")))
        icon = "\u25b6\ufe0f" if event == "started" else "\u23f9\ufe0f"
        text = f"{icon} <b>Orze {event}</b> on <code>{host}</code>\n{msg}"
        return url, {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}

    if event == "heartbeat":
        host = esc(data.get("host", socket.gethostname()))
        lines = [f"\U0001f49a <b>Heartbeat</b> on <code>{host}</code>"]
        lines.append(f"Iter {data.get('iteration', '?')} | "
                      f"Up {esc(str(data.get('uptime', '?')))} | "
                      f"GPUs {data.get('training', 0)}T/{data.get('eval', 0)}E/"
                      f"{data.get('free', 0)}F")
        lines.append(f"Completed {data.get('completed', 0)} | "
                      f"Queued {data.get('queued', 0)} | "
                      f"Failed {data.get('failed', 0)}")
        if data.get("eval_backlog"):
            lines.append(f"Eval backlog: {data['eval_backlog']}")
        if data.get("rate"):
            lines.append(f"Rate: {data['rate']}")
        return url, {"chat_id": chat_id, "text": "\n".join(lines),
                     "parse_mode": "HTML"}

    if event == "milestone":
        text = (f"\U0001f3af <b>Milestone: {esc(str(data.get('count', '?')))} "
                f"experiments completed!</b>")
        return url, {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}

    if event == "disk_warning":
        text = (f"\u26a0\ufe0f <b>Low disk</b> on <code>"
                f"{esc(data.get('host', socket.gethostname()))}</code>\n"
                f"Only {data.get('free_gb', '?')}GB free")
        return url, {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}

    if event == "stall":
        reason = esc(str(data.get("reason", "stalled")))
        text = (f"\U0001f6a8 <b>{reason}</b>: <code>"
                f"{esc(str(data.get('idea_id', '?')))}</code> on GPU "
                f"{data.get('gpu', '?')}")
        return url, {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}

    if event == "role_summary":
        role = esc(str(data.get("role", "?")))
        text = (f"\U0001f9ea <b>{role}</b> finished | "
                f"{data.get('new_ideas', 0)} new ideas | "
                f"{data.get('queued', '?')} queued")
        return url, {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}

    idea_id = esc(str(data.get("idea_id", "")))
    title = esc(str(data.get("title", "")))

    if event == "new_best":
        prev = esc(str(data.get("prev_best_id", "none")))
        metric = esc(str(data.get("metric_name", "")))
        val = esc(str(data.get("metric_value", "")))
        text = (f"<b>NEW BEST</b> <code>{idea_id}</code>: {title}\n"
                f"{metric}: <b>{val}</b>"
                f" (was <code>{prev}</code>)")
    elif event == "failed":
        err = esc(str(data.get("error") or "unknown")[:200])
        text = (f"<b>FAILED</b> <code>{idea_id}</code>: {title}\n"
                f"Error: {err}")
    else:
        t_str = ""
        try:
            t = data.get("training_time")
            if t is not None:
                t_str = f" in {float(t):.0f}s"
        except (ValueError, TypeError):
            pass
        metric = esc(str(data.get("metric_name", "")))
        val = esc(str(data.get("metric_value", "")))
        rank = esc(str(data.get("rank", "?")))
        text = (f"<b>Completed</b> <code>{idea_id}</code>: {title}\n"
                f"{metric}: {val}"
                f" (rank #{rank})"
                f"{t_str}")
    text += _format_leaderboard(data, lambda s: f"<b>{s}</b>", escape_fn=esc)

    return url, {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}


def notify(event: str, data: dict, cfg: dict):
    """Send notifications for an event to all configured channels. Never raises."""
    try:
        ncfg = cfg.get("notifications") or {}
        if not ncfg.get("enabled", False):
            return

        global_on = ncfg.get("on") or ["completed", "failed", "new_best"]
        # Lifecycle/system events always delivered (not filtered)
        lifecycle = {"started", "shutdown", "heartbeat", "milestone",
                     "disk_warning", "stall", "role_summary"}

        for ch in (ncfg.get("channels") or []):
            ch_on = ch.get("on") or global_on
            if event not in lifecycle and event not in ch_on:
                continue

            ch_type = ch.get("type", "webhook")
            if ch_type == "slack":
                _notify_send(ch["webhook_url"], _format_slack(event, data))
            elif ch_type == "discord":
                _notify_send(ch["webhook_url"], _format_discord(event, data))
            elif ch_type == "telegram":
                url, payload = _format_telegram(event, data, ch)
                _notify_send(url, payload)
            elif ch_type == "webhook":
                payload = {"event": event, "data": data,
                           "host": socket.gethostname(),
                           "timestamp": datetime.datetime.now().isoformat()}
                _notify_send(ch["url"], payload, headers=ch.get("headers"))
            else:
                logger.warning("Unknown notification channel: %s", ch_type)
    except Exception as e:
        logger.warning("Notification dispatch error: %s", e)


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def save_state(results_dir: Path, state: dict):
    """Save orchestrator state for restart recovery (per-host)."""
    hostname = socket.gethostname()
    atomic_write(results_dir / f".orze_state_{hostname}.json",
                 json.dumps(state, indent=2))


def load_state(results_dir: Path) -> dict:
    """Load orchestrator state from checkpoint (per-host).
    Falls back to legacy .orze_state.json for backward compat.
    Migrates legacy flat research_* keys into roles dict."""
    hostname = socket.gethostname()
    path = results_dir / f".orze_state_{hostname}.json"

    if not path.exists():
        legacy = results_dir / ".orze_state.json"
        if legacy.exists():
            path = legacy

    if path.exists():
        try:
            state = json.loads(path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError, UnicodeDecodeError):
            logger.warning("Corrupt state file, starting fresh")
            return {"iteration": 0, "failure_counts": {}, "roles": {}}

        # Migrate legacy flat research state into roles dict
        if "roles" not in state and "research_cycles" in state:
            state["roles"] = {
                "research": {
                    "cycles": state.pop("research_cycles", 0),
                    "last_run_time": state.pop("last_research_time", 0.0),
                }
            }
        state.setdefault("roles", {})
        return state
    return {"iteration": 0, "failure_counts": {}, "roles": {}}


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

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

        # Validate config on startup
        config_errors = _validate_config(cfg)
        for err in config_errors:
            logger.error("Config error: %s", err)
        if config_errors:
            raise SystemExit(f"Invalid config: {len(config_errors)} error(s) — see log above")

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

        # Initialize Idea Lake for archival
        try:
            from idea_lake import IdeaLake
            lake_path = Path(cfg.get("ideas_file", "ideas.md")).parent / "idea_lake.db"
            self.lake = IdeaLake(str(lake_path))
            logger.info("Idea Lake initialized: %d archived ideas", self.lake.count())
        except Exception as exc:
            logger.warning("Idea Lake not available: %s", exc)
            self.lake = None

        self.results_dir.mkdir(parents=True, exist_ok=True)

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

    def _shutdown(self, signum, frame):
        """Signal handler — just sets flag so the main loop exits cleanly."""
        if not self.running:
            # Second signal = force exit
            logger.warning("Forced exit (second signal).")
            sys.exit(1)
        logger.info("Received signal %d, will shut down after current step...",
                     signum)
        self.running = False

    def _graceful_shutdown(self):
        """Terminate all child processes, save state, and clean up.

        Called from the main loop after self.running becomes False,
        NOT from the signal handler (avoids re-entrancy issues).
        """
        logger.info("Shutting down gracefully...")

        # 1. Send SIGTERM to all child process groups
        for gpu, tp in self.active.items():
            logger.info("Terminating training %s on GPU %d (PID %d)...",
                        tp.idea_id, gpu, tp.process.pid)
            _kill_pg(tp.process, signal.SIGTERM)
        # Let eval processes continue running (they'll write reports
        # and exit on their own).  The backlog scanner checks for
        # already-running evals before launching duplicates.
        for gpu, ep in self.active_evals.items():
            logger.info("Detaching eval %s on GPU %d (PID %d) "
                        "— will finish in background",
                        ep.idea_id, gpu, ep.process.pid)
        for role_name, rp in self.active_roles.items():
            logger.info("Terminating role '%s' (PID %d)...",
                        role_name, rp.process.pid)
            _kill_pg(rp.process, signal.SIGTERM)

        # 2. Wait for graceful exit (up to 30s), then SIGKILL stragglers
        deadline = time.time() + 30
        for gpu, tp in self.active.items():
            remaining = max(1, deadline - time.time())
            try:
                tp.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                logger.warning("Force killing training %s (PID %d)",
                               tp.idea_id, tp.process.pid)
                _kill_pg(tp.process, signal.SIGKILL)
                try:
                    tp.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    logger.error("Failed to reap training %s", tp.idea_id)
            tp.close_log()
        # Eval processes were detached above — just close our log handles.
        for gpu, ep in self.active_evals.items():
            ep.close_log()
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
                "host": socket.gethostname(),
                "message": (f"Graceful shutdown after iteration "
                            f"{self.iteration}"),
            }, self.cfg)
        except Exception:
            pass

        # 6. Clean up PID file
        self._remove_pid_file()

        logger.info("Shutdown complete. State saved at iteration %d.",
                     self.iteration)

    def _build_machine_status(self) -> list:
        """Build machine status from heartbeats for report notifications."""
        heartbeats = _read_all_heartbeats(self.results_dir)
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
                                    ed, cfg, eval_file)
                            except (json.JSONDecodeError, OSError,
                                    KeyError, UnicodeDecodeError):
                                pass
                    if metric_val is None:
                        # Last resort: use training's internal metric
                        metric_val = deep_get(
                            m, f"test_metrics.{primary}"
                        ) or m.get(primary)
                    if metric_val is None:
                        logger.warning(
                            "Notification for %s has metric_val=None "
                            "(row_pv=%s, m.get(%s)=%s, eval=%s, test_auc=%s)",
                            idea_id,
                            row.get("primary_val"),
                            primary, m.get(primary),
                            (self.results_dir / idea_id /
                             cfg.get("eval_output", "eval_report.json")
                             ).exists(),
                            m.get(primary))
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
                    }, cfg)
                elif status == "FAILED":
                    notify("failed", {
                        "idea_id": idea_id, "title": title,
                        "error": m.get("error", "unknown"),
                        "leaderboard": leaderboard,
                    }, cfg)

                # Auto-archive to Idea Lake
                if self.lake and status in ("COMPLETED", "FAILED"):
                    try:
                        idea_data = ideas.get(idea_id, {})
                        config_yaml = ""
                        raw_md = idea_data.get("raw", "")
                        if idea_data.get("config"):
                            import io
                            config_yaml = yaml.dump(idea_data["config"],
                                                    default_flow_style=False)
                        # Load eval metrics if available
                        eval_metrics = {}
                        eval_file = cfg.get("eval_output", "eval_report.json")
                        eval_path = self.results_dir / idea_id / eval_file
                        if eval_path.exists():
                            try:
                                ed = json.loads(eval_path.read_text())
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
                        self.lake.insert(
                            idea_id, title, config_yaml, raw_md,
                            eval_metrics=eval_metrics or None,
                            status=status.lower(),
                            priority=idea_data.get("priority", "medium"),
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
                r"^## idea-\d+:", ideas_file.read_text(encoding="utf-8"),
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

    def _run_all_roles(self):
        """Check active roles and launch new ones (non-blocking)."""
        # Check active roles
        finished = check_active_roles(
            self.active_roles,
            ideas_file=self.cfg.get("ideas_file", "ideas.md"))
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
                # Role summary notification
                if ideas_modified:
                    ideas_now = parse_ideas(self.cfg["ideas_file"])
                    n_queued = len(get_unclaimed(ideas_now, self.results_dir, set()))
                    prev_count = role_state.get("_prev_idea_count", 0)
                    new_ideas = max(0, len(ideas_now) - prev_count)
                    role_state["_prev_idea_count"] = len(ideas_now)
                    notify("role_summary", {
                        "role": role_name,
                        "new_ideas": new_ideas,
                        "queued": n_queued,
                    }, self.cfg)
            else:
                consec = role_state.get("consecutive_failures", 0) + 1
                role_state["consecutive_failures"] = consec
                if consec >= 3:
                    logger.error("%s has failed %d consecutive times — "
                                 "check config or script", role_name, consec)

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
        # research_agent.py lives next to farm.py in the orze directory
        agent_script = Path(__file__).parent / "research_agent.py"

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
        (train_idea.py, evaluate_dataset.py) with our results dir in their
        cmdline, but whose parent is init (PPID=1) — i.e. orphans.
        Kill their entire process groups.
        """
        my_pid = os.getpid()
        results_str = str(self.results_dir)
        cfg = self.cfg
        patterns = [Path(cfg.get("train_script", "train_idea.py")).name]
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
        """
        stop_file = self.results_dir / ".orze_stop_all"
        if stop_file.exists():
            logger.info("Found .orze_stop_all — shutting down")
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
            msg = disabled_file.read_text().strip()
            logger.error("Orze is DISABLED: %s", msg)
            logger.error("Remove %s to re-enable", disabled_file)
            return True
        return False

    def run(self):
        cfg = self.cfg
        self._write_pid_file()
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

            # 4. Run agent roles (research, documenter, etc.)
            try:
                self._run_all_roles()
            except Exception as e:
                logger.error("Error in _run_all_roles: %s — continuing", e)

            # 5. Parse ideas, expand sweeps, and find unclaimed
            ideas = parse_ideas(cfg["ideas_file"])
            sweep_max = cfg.get("sweep", {}).get("max_combos", 20)
            ideas = expand_sweeps(ideas, max_combos=sweep_max)
            skipped = get_skipped_ideas(
                self.failure_counts,
                cfg.get("max_idea_failures", 0))
            unclaimed = get_unclaimed(ideas, self.results_dir, skipped)

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
                base = tp.idea_id.split("~")[0]
                sweep_counts[base] = sweep_counts.get(base, 0) + 1

            if unclaimed and free and disk_ok:
                for gpu in free:
                    launched = False
                    while unclaimed:
                        idea_id = unclaimed.pop(0)
                        # Enforce per-idea sweep concurrency limit
                        if "~" in idea_id:
                            base = idea_id.split("~")[0]
                            if sweep_counts.get(base, 0) >= max_sweep_concurrent:
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
                        if "~" in idea_id:
                            base = idea_id.split("~")[0]
                            sweep_counts[base] = sweep_counts.get(base, 0) + 1
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

            # 8. Update report
            completed_rows = update_report(self.results_dir, ideas, cfg)

            # 9. Write heartbeat + status.json
            write_host_heartbeat(self.results_dir, self.active, free)
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

            # 9b. Admin cache (pre-aggregated fleet/queue/alerts)
            try:
                write_admin_cache(self.results_dir, ideas, cfg)
            except Exception as e:
                logger.warning("Admin cache write failed: %s", e)

            # 10. Save state
            save_state(self.results_dir, {
                "iteration": self.iteration,
                "failure_counts": self.failure_counts,
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
                        "roles": self.role_states,
                        "best_idea_id": self._best_idea_id,
                    })
                logger.info("Done.")
                break

            time.sleep(cfg["poll"])

        # Main loop exited (signal received or --once finished)
        if self.active or self.active_evals or self.active_roles:
            self._graceful_shutdown()
        else:
            # Nothing running, just save state and clean up
            save_state(self.results_dir, {
                "iteration": self.iteration,
                "failure_counts": self.failure_counts,
                "roles": self.role_states,
                "best_idea_id": self._best_idea_id,
            })
            self._remove_pid_file()
        logger.info("Exited after %d iterations.", self.iteration)


def _count_statuses(ideas: Dict[str, dict], results_dir: Path) -> dict:
    """Count idea statuses without full report generation."""
    counts = {}
    for idea_id in ideas:
        idea_dir = results_dir / idea_id
        if not idea_dir.exists():
            counts["QUEUED"] = counts.get("QUEUED", 0) + 1
        elif (idea_dir / "metrics.json").exists():
            try:
                m = json.loads((idea_dir / "metrics.json").read_text(encoding="utf-8"))
                st = m.get("status", "UNKNOWN")
            except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                st = "FAILED"
            counts[st] = counts.get(st, 0) + 1
        else:
            counts["IN_PROGRESS"] = counts.get("IN_PROGRESS", 0) + 1
    return counts


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False):
    """Configure logging with timestamps."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)


def main():
    parser = argparse.ArgumentParser(
        description="orze: GPU experiment orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python farm.py                             # all GPUs, continuous
  python farm.py -c orze.yaml --gpus 0,1     # with project config
  python farm.py --once                      # one cycle then exit
  python farm.py --report-only               # regenerate report
        """,
    )
    parser.add_argument("-c", "--config-file", type=str, default=None,
                        help="Path to orze.yaml project config")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs (default: auto-detect)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Max training time in seconds")
    parser.add_argument("--poll", type=int, default=None,
                        help="Seconds between iterations")
    parser.add_argument("--once", action="store_true",
                        help="Run one cycle and exit")
    parser.add_argument("--stop", action="store_true",
                        help="Gracefully stop a running orze instance")
    parser.add_argument("--disable", action="store_true",
                        help="Stop and persistently disable Orze (survives restarts)")
    parser.add_argument("--enable", action="store_true",
                        help="Remove persistent disable flag to allow Orze to run")
    parser.add_argument("--report-only", action="store_true",
                        help="Only regenerate report")
    parser.add_argument("--role-only", type=str, default=None, metavar="NAME",
                        help="Run a single agent role once and exit")
    parser.add_argument("--research-only", action="store_true",
                        help="Alias for --role-only research")
    parser.add_argument("--ideas-md", type=str, default=None,
                        help="Path to ideas markdown file")
    parser.add_argument("--base-config", type=str, default=None,
                        help="Path to base config YAML")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory for results")
    parser.add_argument("--train-script", type=str, default=None,
                        help="Training script to run per idea")
    parser.add_argument("--admin", action="store_true",
                        help="Launch admin panel instead of farm loop")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Load project config, then apply CLI overrides
    cfg = load_project_config(args.config_file)
    cfg["_config_path"] = args.config_file  # stored for mode: research

    if args.admin:
        from orze.admin.server import run_admin
        run_admin(cfg)
        return

    if args.timeout is not None:
        cfg["timeout"] = args.timeout
    if args.poll is not None:
        cfg["poll"] = args.poll
    if args.ideas_md:
        cfg["ideas_file"] = args.ideas_md
    if args.base_config:
        cfg["base_config"] = args.base_config
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    if args.train_script:
        cfg["train_script"] = args.train_script

    # Parse GPU list
    if args.gpus:
        gpu_ids = [int(x) for x in args.gpus.split(",")]
    else:
        gpu_ids = detect_all_gpus()
        if not gpu_ids:
            logger.error(
                "No GPUs detected. Use --gpus to specify manually.")
            sys.exit(1)

    # --stop: gracefully stop ALL running orze instances (local + remote)
    if args.stop:
        results_dir = Path(cfg["results_dir"])

        # 1. Write .orze_stop_all — every instance on shared storage
        #    checks this file each iteration and shuts down gracefully
        stop_file = results_dir / ".orze_stop_all"
        stop_file.write_text(
            f"stop requested at {datetime.datetime.now().isoformat()} "
            f"by {socket.gethostname()} PID {os.getpid()}\n",
            encoding="utf-8",
        )
        logger.info("Wrote .orze_stop_all — remote instances will stop "
                     "within one poll cycle (%ds)",
                     cfg.get("poll", 30))

        # 2. Also SIGTERM any local orze process (faster than waiting
        #    for the file check)
        local_pids = set()
        for pid_path in sorted(results_dir.glob(".orze.pid*")):
            try:
                pid = int(pid_path.read_text(encoding="utf-8").strip())
                os.kill(pid, 0)  # check if running locally
                local_pids.add(pid)
            except (ValueError, OSError, ProcessLookupError):
                # Not running locally or stale — clean up
                try:
                    pid_path.unlink(missing_ok=True)
                except OSError:
                    pass

        for pid in local_pids:
            logger.info("Sending SIGTERM to local orze (PID %d)...", pid)
            try:
                os.kill(pid, signal.SIGTERM)
            except (ProcessLookupError, OSError):
                pass

        # 3. Wait for local processes to exit
        if local_pids:
            for _ in range(60):
                still_alive = set()
                for pid in local_pids:
                    try:
                        os.kill(pid, 0)
                        still_alive.add(pid)
                    except ProcessLookupError:
                        logger.info("Local orze (PID %d) stopped.", pid)
                local_pids = still_alive
                if not local_pids:
                    break
                time.sleep(1)
            for pid in local_pids:
                logger.warning("Local orze (PID %d) did not stop within 60s. "
                               "Send SIGKILL with: kill -9 %d", pid, pid)

        # NOTE: .orze_stop_all is NOT deleted here — remote instances
        # need time to see it. It gets cleaned up on next run() start.
        logger.info("Stop complete. Remote instances will stop within "
                     "one poll cycle.")
        return

    # --disable: stop + persistently prevent Orze from starting
    if args.disable:
        results_dir = Path(cfg["results_dir"])
        disabled_file = results_dir / ".orze_disabled"
        disabled_file.write_text(
            f"disabled at {datetime.datetime.now().isoformat()} "
            f"by {socket.gethostname()} PID {os.getpid()}\n",
            encoding="utf-8",
        )
        logger.info("Wrote .orze_disabled — Orze will refuse to start on any machine")
        # Also trigger a normal stop for any currently running instances
        stop_file = results_dir / ".orze_stop_all"
        stop_file.write_text(
            f"stop requested at {datetime.datetime.now().isoformat()} "
            f"by {socket.gethostname()} PID {os.getpid()}\n",
            encoding="utf-8",
        )
        logger.info("Also wrote .orze_stop_all for running instances")
        logger.info("To re-enable: python orze/farm.py -c orze.yaml --enable")
        return

    # --enable: remove persistent disable flag
    if args.enable:
        results_dir = Path(cfg["results_dir"])
        disabled_file = results_dir / ".orze_disabled"
        if disabled_file.exists():
            disabled_file.unlink()
            logger.info("Removed .orze_disabled — Orze can now start")
        else:
            logger.info("Orze was not disabled (no .orze_disabled found)")
        return

    # Report-only mode
    if args.report_only:
        ideas = parse_ideas(cfg["ideas_file"])
        update_report(Path(cfg["results_dir"]), ideas, cfg)
        return

    # Role-only mode (--role-only NAME or --research-only)
    role_only = args.role_only
    if args.research_only:
        role_only = "research"
    if role_only:
        roles_cfg = cfg.get("roles") or {}
        role_cfg = roles_cfg.get(role_only)
        if not role_cfg or not isinstance(role_cfg, dict):
            logger.error("No role '%s' configured in orze.yaml", role_only)
            sys.exit(1)
        mode = role_cfg.get("mode", "script")
        if mode == "claude":
            has_target = role_cfg.get("rules_file")
        elif mode == "research":
            has_target = role_cfg.get("backend")
        else:
            has_target = role_cfg.get("script")
        if not has_target:
            target_key = {"claude": "rules_file", "research": "backend"}.get(mode, "script")
            logger.error("No %s.%s configured in orze.yaml", role_only, target_key)
            sys.exit(1)
        orze = Orze(gpu_ids, cfg, once=True)
        # Force immediate run by zeroing cooldown
        rs = orze.role_states.setdefault(
            role_only, {"cycles": 0, "last_run_time": 0.0})
        rs["last_run_time"] = 0.0
        orze._run_role_step(role_only, role_cfg)
        save_state(orze.results_dir, {
            "iteration": orze.iteration,
            "failure_counts": orze.failure_counts,
            "fix_counts": orze.fix_counts,
            "roles": orze.role_states,
            "best_idea_id": orze._best_idea_id,
        })
        return

    orze = Orze(gpu_ids, cfg, once=args.once)
    orze.run()


if __name__ == "__main__":
    main()
