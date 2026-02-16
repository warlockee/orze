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

import yaml

__version__ = "0.4.0"

logger = logging.getLogger("orze")


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
        logger.info("Loaded config from %s", path)
    elif path:
        logger.warning("Config file %s not found, using defaults", path)

    # Migrate legacy research: into roles: dict
    if "research" in cfg and isinstance(cfg["research"], dict):
        if not cfg.get("roles"):
            cfg["roles"] = {"research": cfg["research"]}
        elif "research" not in cfg["roles"]:
            cfg["roles"]["research"] = cfg["research"]

    return cfg


# ---------------------------------------------------------------------------
# Ideas parsing
# ---------------------------------------------------------------------------

PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def parse_ideas(path: str) -> Dict[str, dict]:
    """Parse ideas.md into {idea_id: {title, priority, config, raw}}."""
    try:
        text = Path(path).read_text(encoding="utf-8")
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

        ideas[idea_id] = {
            "title": title,
            "priority": priority,
            "config": config,
            "raw": raw.strip(),
        }
    return ideas


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


def detect_oom(tp: TrainingProcess) -> bool:
    """Check log tail for OOM indicators."""
    tail = tail_file(tp.log_path, 4096)
    oom_patterns = [
        "CUDA out of memory",
        "OutOfMemoryError",
        "torch.cuda.OutOfMemoryError",
        "CUDA error: out of memory",
    ]
    return any(p in tail for p in oom_patterns)


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

def check_active(active: Dict[int, TrainingProcess], results_dir: Path,
                 cfg: dict, failure_counts: dict) -> list:
    """Check running processes. Reap completed/timed-out/stalled/OOM.
    Returns list of (idea_id, gpu) tuples for finished ideas."""
    finished = []
    stall_minutes = cfg.get("stall_minutes", 0)

    for gpu in list(active.keys()):
        tp = active[gpu]
        ret = tp.process.poll()
        elapsed = time.time() - tp.start_time

        # --- Still running ---
        if ret is None:
            if elapsed > tp.timeout:
                logger.warning("[TIMEOUT] %s after %.0fm — killing",
                               tp.idea_id, elapsed / 60)
                tp.process.terminate()
                try:
                    tp.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    tp.process.kill()
                    tp.process.wait()
                tp.close_log()
                _write_failure(results_dir / tp.idea_id, "Timed out")
                _record_failure(failure_counts, tp.idea_id)
                del active[gpu]
                finished.append((tp.idea_id, gpu))
                continue

            if check_stalled(tp, stall_minutes):
                logger.warning("[STALLED] %s — no log output for %dm, killing",
                               tp.idea_id, stall_minutes)
                tp.process.terminate()
                try:
                    tp.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    tp.process.kill()
                    tp.process.wait()
                tp.close_log()
                _write_failure(results_dir / tp.idea_id,
                               f"Stalled (no output for {stall_minutes}m)")
                _record_failure(failure_counts, tp.idea_id)
                del active[gpu]
                finished.append((tp.idea_id, gpu))
                continue

            if detect_oom(tp) and tp.process.poll() is None:
                logger.warning("[OOM] %s — CUDA out of memory detected, killing",
                               tp.idea_id)
                tp.process.terminate()
                try:
                    tp.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    tp.process.kill()
                    tp.process.wait()
                tp.close_log()
                _write_failure(results_dir / tp.idea_id, "CUDA out of memory")
                _record_failure(failure_counts, tp.idea_id)
                del active[gpu]
                finished.append((tp.idea_id, gpu))
                continue

            continue

        # --- Process exited ---
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
    try:
        ep.process.wait(timeout=ep.timeout)
        if ep.process.returncode == 0:
            logger.info("Eval completed for %s", idea_id)
        else:
            logger.warning("Eval failed for %s (exit %d)",
                           idea_id, ep.process.returncode)
    except subprocess.TimeoutExpired:
        logger.warning("Eval timed out for %s after %ds",
                       idea_id, ep.timeout)
        ep.process.terminate()
        try:
            ep.process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            ep.process.kill()
            ep.process.wait()
    except Exception as e:
        logger.warning("Eval error for %s: %s", idea_id, e)
    finally:
        ep.close_log()


def check_active_evals(active_evals: Dict[int, EvalProcess],
                       results_dir: Path, cfg: dict) -> list:
    """Check running eval processes. Returns list of (idea_id, gpu) for finished evals."""
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
                ep.process.terminate()
                try:
                    ep.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    ep.process.kill()
                    ep.process.wait()
                ep.close_log()
                del active_evals[gpu]
                finished.append((ep.idea_id, gpu))
            continue

        # Process exited
        ep.close_log()
        if ret == 0:
            logger.info("[EVAL OK] %s on GPU %d in %.1fm",
                        ep.idea_id, gpu, elapsed / 60)
        else:
            logger.warning("[EVAL FAILED] %s on GPU %d — exit %d",
                           ep.idea_id, gpu, ret)
        del active_evals[gpu]
        finished.append((ep.idea_id, gpu))

    return finished


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
    """Run periodic cleanup: delete files matching patterns, run cleanup script."""
    cleanup_cfg = cfg.get("cleanup") or {}

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

    for idea_id in sorted(ideas.keys(), key=_id_sort_key):
        idea_dir = results_dir / idea_id

        if not idea_dir.exists():
            rows.append({"id": idea_id, "title": ideas[idea_id]["title"],
                         "status": "QUEUED", "values": {}})
            continue

        metrics_path = idea_dir / "metrics.json"
        if not metrics_path.exists():
            rows.append({"id": idea_id, "title": ideas[idea_id]["title"],
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
            "id": idea_id, "title": ideas[idea_id]["title"],
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

    if completed:
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

        for rank, r in enumerate(completed, 1):
            row = f"| {rank} | {r['id']} | {r['title'][:40]}"
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
            pri = ideas[r["id"]]["priority"]
            lines.append(f"- **{r['id']}** [{pri}]: {r['title'][:60]}")
        if len(queued) > 20:
            lines.append(f"- ... and {len(queued) - 20} more")
        lines.append("")

    report_path = results_dir / "report.md"
    atomic_write(report_path, "\n".join(lines))
    logger.info("Report updated: %d completed, %d queued, %d failed",
                counts.get("COMPLETED", 0), counts.get("QUEUED", 0),
                counts.get("FAILED", 0))

    return completed


# ---------------------------------------------------------------------------
# Status JSON (for LLM consumption)
# ---------------------------------------------------------------------------

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
        text = f"```\n{_format_report_text(data)}\n```"
        return {"text": text}
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
        try:
            t = float(data.get("training_time") or 0)
        except (ValueError, TypeError):
            t = 0.0
        text = (f":white_check_mark: *Completed* `{data['idea_id']}`: "
                f"{data['title']}\n"
                f"{data['metric_name']}: {data['metric_value']}"
                f" (rank #{data.get('rank', '?')})"
                f" in {t:.0f}s")
    text += _format_leaderboard(data, lambda s: f"*{s}*")
    return {"text": text}


def _format_discord(event: str, data: dict) -> dict:
    """Format notification for Discord webhook."""
    if event == "report":
        return {"content": f"```\n{_format_report_text(data)}\n```"}
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
        try:
            t = float(data.get("training_time") or 0)
        except (ValueError, TypeError):
            t = 0.0
        content = (f"**Completed** `{data['idea_id']}`: {data['title']}\n"
                   f"{data['metric_name']}: {data['metric_value']}"
                   f" (rank #{data.get('rank', '?')})"
                   f" in {t:.0f}s")
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
        try:
            t = float(data.get("training_time") or 0)
        except (ValueError, TypeError):
            t = 0.0
        metric = esc(str(data.get("metric_name", "")))
        val = esc(str(data.get("metric_value", "")))
        rank = esc(str(data.get("rank", "?")))
        text = (f"<b>Completed</b> <code>{idea_id}</code>: {title}\n"
                f"{metric}: {val}"
                f" (rank #{rank})"
                f" in {t:.0f}s")
    text += _format_leaderboard(data, lambda s: f"<b>{s}</b>", escape_fn=esc)

    return url, {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}


def notify(event: str, data: dict, cfg: dict):
    """Send notifications for an event to all configured channels. Never raises."""
    try:
        ncfg = cfg.get("notifications") or {}
        if not ncfg.get("enabled", False):
            return

        global_on = ncfg.get("on") or ["completed", "failed", "new_best"]

        for ch in (ncfg.get("channels") or []):
            ch_on = ch.get("on") or global_on
            if event not in ch_on:
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
        self.running = True

        state = load_state(self.results_dir)
        self.iteration = state.get("iteration", 0)
        self.failure_counts = state.get("failure_counts", {})

        # Per-role agent state: {role_name: {"cycles": int, "last_run_time": float}}
        self.role_states: Dict[str, dict] = state.get("roles", {})
        self._best_idea_id: Optional[str] = state.get("best_idea_id")

        self.results_dir.mkdir(parents=True, exist_ok=True)

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        logger.info("Received signal %d, shutting down gracefully...", signum)
        self.running = False
        # Terminate all training processes first
        for gpu, tp in self.active.items():
            logger.info("Terminating %s on GPU %d...", tp.idea_id, gpu)
            tp.process.terminate()
        # Terminate all eval processes
        for gpu, ep in self.active_evals.items():
            logger.info("Terminating eval %s on GPU %d...", ep.idea_id, gpu)
            ep.process.terminate()
        deadline = time.time() + 300
        for gpu, tp in self.active.items():
            remaining = max(1, deadline - time.time())
            try:
                tp.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                logger.warning("Force killing %s", tp.idea_id)
                tp.process.kill()
                tp.process.wait()
            tp.close_log()
        for gpu, ep in self.active_evals.items():
            remaining = max(1, deadline - time.time())
            try:
                ep.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                ep.process.kill()
                ep.process.wait()
            ep.close_log()
        save_state(self.results_dir, {
            "iteration": self.iteration,
            "failure_counts": self.failure_counts,
            "roles": self.role_states,
            "best_idea_id": self._best_idea_id,
        })
        logger.info("Shutdown complete.")
        sys.exit(0)

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
                return

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
                    # Use primary_val from report rows (reads from
                    # nexar_test_report.json etc.), fall back to metrics.json
                    row = row_lookup.get(idea_id, {})
                    metric_val = row.get("primary_val") or m.get(primary)
                    t_time = m.get("training_time", 0)
                    if not t_time:
                        t_time = m.get("training_log_summary", {}).get(
                            "total_training_time", 0)
                    notify("completed", {
                        "idea_id": idea_id, "title": title,
                        "metric_name": primary,
                        "metric_value": metric_val,
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

            # New best detection
            if completed_rows:
                current_best = completed_rows[0]["id"]
                if (self._best_idea_id is not None
                        and current_best != self._best_idea_id):
                    notify("new_best", {
                        "idea_id": current_best,
                        "title": completed_rows[0]["title"],
                        "metric_name": primary,
                        "metric_value": completed_rows[0].get("primary_val"),
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
        """Run an agent role (rate-limited, locked, logged).

        Supports two modes:
          - mode: script  — run a Python script
          - mode: claude  — run Claude CLI with a rules/prompt file
        """
        mode = role_cfg.get("mode", "script")
        if mode == "script" and not role_cfg.get("script"):
            return
        if mode == "claude" and not role_cfg.get("rules_file"):
            return

        # Per-role cooldown
        role_state = self.role_states.setdefault(
            role_name, {"cycles": 0, "last_run_time": 0.0})
        cooldown = role_cfg.get("cooldown", 300)
        elapsed = time.time() - role_state["last_run_time"]
        if elapsed < cooldown:
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

        try:
            with open(log_path, "w", encoding="utf-8") as log_fh:
                result = subprocess.run(
                    cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
                    timeout=timeout,
                )

            role_state["last_run_time"] = time.time()
            role_state["cycles"] += 1

            if result.returncode == 0:
                logger.info("%s cycle %d completed", role_name, cycle_num)
            else:
                logger.warning(
                    "%s failed (exit %d), see %s",
                    role_name, result.returncode, log_path)

        except subprocess.TimeoutExpired:
            role_state["last_run_time"] = time.time()
            role_state["cycles"] += 1
            logger.warning("%s timed out after %ds", role_name, timeout)
        except Exception as e:
            role_state["last_run_time"] = time.time()
            logger.warning("%s error: %s", role_name, e)
        finally:
            _fs_unlock(lock_dir)

    def _run_all_roles(self):
        """Run all configured agent roles (each independently rate-limited)."""
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

        # --allowedTools (default: let Claude read/write files)
        allowed_tools = research_cfg.get("allowed_tools") or "Read,Write,Edit,Glob,Grep,Bash"
        cmd.extend(["--allowedTools", str(allowed_tools)])

        # --output-format
        output_format = research_cfg.get("output_format") or "text"
        cmd.extend(["--output-format", str(output_format)])

        # Any extra CLI args
        cmd.extend(_format_args(research_cfg.get("claude_args") or [],
                                template_vars))

        return cmd

    @staticmethod
    def _count_new_ideas(log_path: Path) -> int:
        """Parse research log to count how many new ideas were generated."""
        try:
            log_text = log_path.read_text(encoding="utf-8")
            for line in log_text.split("\n"):
                if "ideas" in line.lower():
                    nums = re.findall(
                        r"(\d+)\s+(?:new\s+)?ideas?", line, re.I)
                    if nums:
                        return int(nums[-1])
        except Exception:
            pass
        return 0

    def run(self):
        cfg = self.cfg
        logger.info("Starting orze v%s on GPUs %s", __version__, self.gpu_ids)
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

        while self.running:
            self.iteration += 1
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            logger.info("--- Iteration %d [%s] ---", self.iteration, ts)

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
                                        cfg, self.failure_counts)

            # 3a. Check active eval processes
            eval_finished = []
            if self.active_evals:
                eval_finished = check_active_evals(
                    self.active_evals, self.results_dir, cfg)

            # 4. Launch evals for newly completed training (non-blocking)
            for idea_id, gpu in finished:
                metrics_path = self.results_dir / idea_id / "metrics.json"
                if metrics_path.exists():
                    try:
                        metrics = json.loads(
                            metrics_path.read_text(encoding="utf-8"))
                        if metrics.get("status") == "COMPLETED":
                            ep = launch_eval(
                                idea_id, gpu, self.results_dir, cfg)
                            if ep is not None:
                                self.active_evals[gpu] = ep
                            else:
                                # No eval configured or already done
                                eval_finished.append((idea_id, gpu))
                        else:
                            eval_finished.append((idea_id, gpu))
                    except (json.JSONDecodeError, OSError, UnicodeDecodeError):
                        eval_finished.append((idea_id, gpu))
                else:
                    eval_finished.append((idea_id, gpu))

            # 4a. Run post-scripts for evals that just completed
            for idea_id, gpu in eval_finished:
                run_post_scripts(idea_id, gpu, self.results_dir, cfg)

            # 5. Run agent roles (research, documenter, etc.)
            self._run_all_roles()

            # 6. Parse ideas and find unclaimed
            ideas = parse_ideas(cfg["ideas_file"])
            skipped = get_skipped_ideas(
                self.failure_counts,
                cfg.get("max_idea_failures", 0))
            unclaimed = get_unclaimed(ideas, self.results_dir, skipped)

            # 7. Find free GPUs and launch training
            #    A GPU is free if not running training AND not running eval
            busy_gpus = set(self.active.keys()) | set(self.active_evals.keys())
            free = [g for g in self.gpu_ids if g not in busy_gpus
                    and (get_gpu_memory_used(g) or 0)
                    <= cfg.get("gpu_mem_threshold", 2000)]

            if unclaimed and free and disk_ok:
                for gpu in free:
                    launched = False
                    while unclaimed:
                        idea_id = unclaimed.pop(0)
                        if not claim(idea_id, self.results_dir, gpu):
                            # Another machine claimed it, try next idea
                            continue
                        # Run pre-script (e.g., feature extraction check)
                        if not run_pre_script(idea_id, gpu, cfg):
                            logger.warning(
                                "Pre-script failed for %s, marking FAILED",
                                idea_id)
                            _write_failure(
                                self.results_dir / idea_id,
                                "Pre-script failed")
                            _record_failure(self.failure_counts, idea_id)
                            continue
                        logger.info("Launching %s on GPU %d: %s",
                                    idea_id, gpu,
                                    ideas[idea_id]["title"][:50])
                        try:
                            tp = launch(idea_id, gpu, self.results_dir, cfg)
                        except Exception as e:
                            logger.error("Failed to launch %s on GPU %d: %s",
                                         idea_id, gpu, e)
                            _write_failure(self.results_dir / idea_id,
                                           f"Launch error: {e}")
                            _record_failure(self.failure_counts, idea_id)
                            continue
                        self.active[gpu] = tp
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
                            cfg, self.failure_counts)
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
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Debug logging")
    args = parser.parse_args()

    setup_logging(args.verbose)

    # Load project config, then apply CLI overrides
    cfg = load_project_config(args.config_file)

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
        has_target = (role_cfg.get("rules_file") if mode == "claude"
                      else role_cfg.get("script"))
        if not has_target:
            logger.error("No %s.%s configured in orze.yaml",
                         role_only,
                         "rules_file" if mode == "claude" else "script")
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
            "roles": orze.role_states,
            "best_idea_id": orze._best_idea_id,
        })
        return

    orze = Orze(gpu_ids, cfg, once=args.once)
    orze.run()


if __name__ == "__main__":
    main()
