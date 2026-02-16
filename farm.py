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
"""

import argparse
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
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

__version__ = "0.3.0"

logger = logging.getLogger("orze")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def atomic_write(path: Path, content: str):
    """Write content atomically via tmp+rename (POSIX atomic)."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(content)
    tmp.rename(path)


def tail_file(path: Path, n_bytes: int = 4096) -> str:
    """Read last n_bytes of a file."""
    try:
        size = path.stat().st_size
        with open(path, "r", errors="replace") as f:
            f.seek(max(0, size - n_bytes))
            return f.read()
    except Exception:
        return ""


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
}


def load_project_config(path: Optional[str]) -> dict:
    """Load orze.yaml and merge with defaults. Returns full config dict."""
    cfg = dict(DEFAULT_CONFIG)
    cfg["report"] = dict(DEFAULT_CONFIG["report"])

    if path and Path(path).exists():
        raw = yaml.safe_load(Path(path).read_text()) or {}
        for k, v in raw.items():
            if k == "report" and isinstance(v, dict):
                cfg["report"] = {**cfg["report"], **v}
            else:
                cfg[k] = v
        logger.info("Loaded config from %s", path)
    elif path:
        logger.warning("Config file %s not found, using defaults", path)

    return cfg


# ---------------------------------------------------------------------------
# Ideas parsing
# ---------------------------------------------------------------------------

PRIORITY_ORDER = {"critical": 0, "high": 1, "medium": 2, "low": 3}


def parse_ideas(path: str) -> Dict[str, dict]:
    """Parse ideas.md into {idea_id: {title, priority, config, raw}}."""
    text = Path(path).read_text()
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
        num = int(re.search(r"\d+", idea_id).group())
        return (pri, num)

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


def run_pre_script(idea_id: str, gpu: int, cfg: dict) -> bool:
    """Run pre-training script if configured. Returns True if OK to proceed."""
    pre_script = cfg.get("pre_script")
    if not pre_script:
        return True

    python = cfg.get("python", sys.executable)
    pre_args = cfg.get("pre_args", [])
    pre_timeout = cfg.get("pre_timeout", 3600)

    cmd = [python, pre_script]
    for arg in pre_args:
        cmd.append(str(arg).format(idea_id=idea_id, gpu=gpu))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for k, v in cfg.get("train_extra_env", {}).items():
        env[k] = str(v)

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
    for arg in cfg.get("train_extra_args", []):
        cmd.append(str(arg))

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    for k, v in cfg.get("train_extra_env", {}).items():
        env[k] = str(v)

    # Keep file handle open for subprocess lifetime (fixes FD leak)
    log_fh = open(log_path, "w")
    proc = subprocess.Popen(
        cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
    )

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
    except FileNotFoundError:
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
                claim_time = claim_path.stat().st_mtime
                if claim_time < cutoff:
                    shutil.rmtree(d)
                    logger.info("Cleaned orphan: %s (claimed %.1fh ago)",
                                d.name, (time.time() - claim_time) / 3600)
                    cleaned += 1
            except Exception as e:
                logger.warning("Failed to clean orphan %s: %s", d.name, e)

    return cleaned


# ---------------------------------------------------------------------------
# Process checking (with health monitoring)
# ---------------------------------------------------------------------------

def check_active(active: Dict[int, TrainingProcess], results_dir: Path,
                 cfg: dict, failure_counts: dict) -> List[str]:
    """Check running processes. Reap completed/timed-out/stalled/OOM.
    Returns list of finished idea IDs."""
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
                tp.process.kill()
                tp.process.wait()
                tp.close_log()
                _write_failure(results_dir / tp.idea_id, "Timed out")
                _record_failure(failure_counts, tp.idea_id)
                del active[gpu]
                finished.append(tp.idea_id)
                continue

            if check_stalled(tp, stall_minutes):
                logger.warning("[STALLED] %s — no log output for %dm, killing",
                               tp.idea_id, stall_minutes)
                tp.process.kill()
                tp.process.wait()
                tp.close_log()
                _write_failure(results_dir / tp.idea_id,
                               f"Stalled (no output for {stall_minutes}m)")
                _record_failure(failure_counts, tp.idea_id)
                del active[gpu]
                finished.append(tp.idea_id)
                continue

            if detect_oom(tp) and tp.process.poll() is None:
                logger.warning("[OOM] %s — CUDA out of memory detected, killing",
                               tp.idea_id)
                tp.process.kill()
                tp.process.wait()
                tp.close_log()
                _write_failure(results_dir / tp.idea_id, "CUDA out of memory")
                _record_failure(failure_counts, tp.idea_id)
                del active[gpu]
                finished.append(tp.idea_id)
                continue

            continue

        # --- Process exited ---
        tp.close_log()
        metrics_path = results_dir / tp.idea_id / "metrics.json"

        if ret == 0 and metrics_path.exists():
            try:
                metrics = json.loads(metrics_path.read_text())
            except json.JSONDecodeError:
                metrics = {"status": "UNKNOWN"}
            status = metrics.get("status", "COMPLETED")
            logger.info("[%s] %s on GPU %d in %.1fm",
                        status, tp.idea_id, gpu, elapsed / 60)
            if status == "FAILED":
                _record_failure(failure_counts, tp.idea_id)
        else:
            reason = f"exit code {ret}"
            try:
                lines = tp.log_path.read_text().strip().split("\n")
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
        finished.append(tp.idea_id)

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

def run_eval(idea_id: str, gpu: int, results_dir: Path, cfg: dict):
    """Run post-training evaluation if configured."""
    eval_script = cfg.get("eval_script")
    if not eval_script:
        return

    eval_output = cfg.get("eval_output", "eval_report.json")
    output_path = results_dir / idea_id / eval_output
    if output_path.exists():
        logger.debug("Eval already exists for %s, skipping", idea_id)
        return

    metrics_path = results_dir / idea_id / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            if metrics.get("status") != "COMPLETED":
                return
        except json.JSONDecodeError:
            return
    else:
        return

    python = cfg.get("python", sys.executable)
    eval_args = cfg.get("eval_args", [])
    eval_timeout = cfg.get("eval_timeout", 3600)

    cmd = [python, eval_script]
    for arg in eval_args:
        cmd.append(str(arg).format(idea_id=idea_id, gpu=gpu))

    log_path = results_dir / idea_id / "eval_output.log"
    logger.info("Running eval for %s on GPU %d", idea_id, gpu)

    try:
        with open(log_path, "w") as log_fh:
            env = os.environ.copy()
            for k, v in cfg.get("train_extra_env", {}).items():
                env[k] = str(v)
            result = subprocess.run(
                cmd, env=env, stdout=log_fh, stderr=subprocess.STDOUT,
                timeout=eval_timeout,
            )
        if result.returncode == 0:
            logger.info("Eval completed for %s", idea_id)
        else:
            logger.warning("Eval failed for %s (exit %d)",
                           idea_id, result.returncode)
    except subprocess.TimeoutExpired:
        logger.warning("Eval timed out for %s after %ds",
                       idea_id, eval_timeout)
    except Exception as e:
        logger.warning("Eval error for %s: %s", idea_id, e)


def run_post_scripts(idea_id: str, gpu: int, results_dir: Path, cfg: dict):
    """Run additional post-training scripts (beyond eval_script).
    Each entry in post_scripts is a dict with: script, args, timeout, output."""
    post_scripts = cfg.get("post_scripts", [])
    if not post_scripts:
        return

    # Check training succeeded
    metrics_path = results_dir / idea_id / "metrics.json"
    if metrics_path.exists():
        try:
            metrics = json.loads(metrics_path.read_text())
            if metrics.get("status") != "COMPLETED":
                return
        except json.JSONDecodeError:
            return
    else:
        return

    python = cfg.get("python", sys.executable)
    env = os.environ.copy()
    for k, v in cfg.get("train_extra_env", {}).items():
        env[k] = str(v)

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

        args = ps.get("args", [])
        timeout = ps.get("timeout", 3600)
        name = ps.get("name", f"post-script-{i}")

        cmd = [python, script]
        for arg in args:
            cmd.append(str(arg).format(idea_id=idea_id, gpu=gpu))

        log_path = results_dir / idea_id / f"{name}.log"
        logger.info("Running %s for %s", name, idea_id)

        try:
            with open(log_path, "w") as log_fh:
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
    cleanup_cfg = cfg.get("cleanup", {})

    # Built-in: delete files matching glob patterns in results dirs
    patterns = cleanup_cfg.get("patterns", [])
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
    key = col["key"]

    if source and ":" in source:
        filename, dotpath = source.split(":", 1)
        filepath = results_dir / idea_id / filename
        if filepath.exists():
            try:
                data = json.loads(filepath.read_text())
                return deep_get(data, dotpath)
            except (json.JSONDecodeError, KeyError):
                return None
        return None

    metrics_path = results_dir / idea_id / "metrics.json"
    if metrics_path.exists():
        try:
            data = json.loads(metrics_path.read_text())
            return deep_get(data, key) if "." in key else data.get(key)
        except json.JSONDecodeError:
            return None
    return None


def update_report(results_dir: Path, ideas: Dict[str, dict],
                  cfg: dict) -> list:
    """Generate a configurable leaderboard report.md from all results.
    Returns sorted list of completed row dicts."""
    report_cfg = cfg.get("report", DEFAULT_CONFIG["report"])
    primary_metric = report_cfg.get("primary_metric", "test_accuracy")
    sort_order = report_cfg.get("sort", "descending")
    columns = report_cfg.get("columns", DEFAULT_CONFIG["report"]["columns"])
    title = report_cfg.get("title", "Orze Report")

    rows = []
    for idea_id in sorted(ideas.keys(),
                          key=lambda x: int(re.search(r"\d+", x).group())):
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
            metrics = json.loads(metrics_path.read_text())
        except json.JSONDecodeError:
            metrics = {"status": "FAILED", "error": "corrupt metrics.json"}

        values = {}
        for col in columns:
            values[col["key"]] = _read_metric_value(results_dir, idea_id, col)

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
    completed.sort(key=lambda r: r.get("primary_val") or 0, reverse=reverse)

    if completed:
        header = "| Rank | Idea | Title"
        sep = "|------|------|------"
        for col in columns:
            header += f" | {col['label']}"
            sep += " |" + "-" * max(6, len(col["label"]))
        header += " |"
        sep += " |"
        lines.append(header)
        lines.append(sep)

        for rank, r in enumerate(completed, 1):
            row = f"| {rank} | {r['id']} | {r['title'][:40]}"
            for col in columns:
                val = r["values"].get(col["key"])
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

def write_status_json(results_dir: Path, iteration: int,
                      active: Dict[int, TrainingProcess],
                      free_gpus: List[int], queue_depth: int,
                      completed_count: int, failed_count: int,
                      skipped_count: int, top_results: list,
                      cfg: dict):
    """Write machine-readable status.json for LLM agents."""
    disk_free_gb = 0.0
    try:
        usage = shutil.disk_usage(results_dir)
        disk_free_gb = usage.free / (1024 ** 3)
    except Exception:
        pass

    status = {
        "timestamp": datetime.datetime.now().isoformat(),
        "iteration": iteration,
        "active": [
            {
                "idea_id": tp.idea_id,
                "gpu": tp.gpu,
                "elapsed_min": round((time.time() - tp.start_time) / 60, 1),
            }
            for tp in active.values()
        ],
        "free_gpus": free_gpus,
        "queue_depth": queue_depth,
        "completed": completed_count,
        "failed": failed_count,
        "skipped": skipped_count,
        "disk_free_gb": round(disk_free_gb, 1),
        "top_results": top_results[:10],
    }

    atomic_write(results_dir / "status.json", json.dumps(status, indent=2))


# ---------------------------------------------------------------------------
# State persistence
# ---------------------------------------------------------------------------

def save_state(results_dir: Path, state: dict):
    """Save orchestrator state for restart recovery."""
    atomic_write(results_dir / ".orze_state.json",
                 json.dumps(state, indent=2))


def load_state(results_dir: Path) -> dict:
    """Load orchestrator state from checkpoint."""
    path = results_dir / ".orze_state.json"
    if path.exists():
        try:
            return json.loads(path.read_text())
        except json.JSONDecodeError:
            logger.warning("Corrupt state file, starting fresh")
    return {"iteration": 0, "failure_counts": {}}


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
        self.running = True

        state = load_state(self.results_dir)
        self.iteration = state.get("iteration", 0)
        self.failure_counts = state.get("failure_counts", {})

        self.results_dir.mkdir(parents=True, exist_ok=True)

        signal.signal(signal.SIGINT, self._shutdown)
        signal.signal(signal.SIGTERM, self._shutdown)

    def _shutdown(self, signum, frame):
        logger.info("Received signal %d, shutting down gracefully...", signum)
        self.running = False
        deadline = time.time() + 300
        for gpu, tp in self.active.items():
            logger.info("Waiting for %s on GPU %d...", tp.idea_id, gpu)
            remaining = max(1, deadline - time.time())
            try:
                tp.process.wait(timeout=remaining)
            except subprocess.TimeoutExpired:
                logger.warning("Force killing %s", tp.idea_id)
                tp.process.kill()
            tp.close_log()
        save_state(self.results_dir, {
            "iteration": self.iteration,
            "failure_counts": self.failure_counts,
        })
        logger.info("Shutdown complete.")
        sys.exit(0)

    def run(self):
        cfg = self.cfg
        logger.info("Starting orze v%s on GPUs %s", __version__, self.gpu_ids)
        logger.info("Ideas: %s | Results: %s | Timeout: %ds | Poll: %ds",
                     cfg["ideas_file"], cfg["results_dir"],
                     cfg["timeout"], cfg["poll"])

        while self.running:
            self.iteration += 1
            ts = datetime.datetime.now().strftime("%H:%M:%S")
            logger.info("--- Iteration %d [%s] ---", self.iteration, ts)

            # 1. Check disk space
            if not check_disk_space(self.results_dir,
                                    cfg.get("min_disk_gb", 0)):
                logger.warning(
                    "Low disk space (< %dGB free). Pausing launches.",
                    cfg["min_disk_gb"])
                time.sleep(cfg["poll"])
                continue

            # 2. Periodic maintenance (orphans + GC)
            cleanup_cfg = cfg.get("cleanup", {})
            cleanup_interval = cleanup_cfg.get("interval", 100)
            if cleanup_interval > 0 and self.iteration % cleanup_interval == 0:
                orphan_hours = cfg.get("orphan_timeout_hours", 0)
                if orphan_hours > 0:
                    cleaned = cleanup_orphans(self.results_dir, orphan_hours)
                    if cleaned:
                        logger.info("Cleaned %d orphaned claims", cleaned)
                run_cleanup(self.results_dir, cfg)

            # 3. Check active processes (with health monitoring)
            finished = []
            if self.active:
                finished = check_active(self.active, self.results_dir,
                                        cfg, self.failure_counts)

            # 4. Run post-training steps for newly completed ideas
            for idea_id in finished:
                metrics_path = self.results_dir / idea_id / "metrics.json"
                if metrics_path.exists():
                    try:
                        metrics = json.loads(metrics_path.read_text())
                        if metrics.get("status") == "COMPLETED":
                            run_eval(idea_id, 0, self.results_dir, cfg)
                            run_post_scripts(idea_id, 0, self.results_dir,
                                             cfg)
                    except json.JSONDecodeError:
                        pass

            # 5. Parse ideas and find unclaimed
            ideas = parse_ideas(cfg["ideas_file"])
            skipped = get_skipped_ideas(
                self.failure_counts,
                cfg.get("max_idea_failures", 0))
            unclaimed = get_unclaimed(ideas, self.results_dir, skipped)

            # 6. Find free GPUs and launch
            free = get_free_gpus(self.gpu_ids, self.active,
                                 cfg.get("gpu_mem_threshold", 2000))

            if unclaimed and free:
                for gpu in free:
                    if not unclaimed:
                        break
                    idea_id = unclaimed[0]
                    if claim(idea_id, self.results_dir, gpu):
                        # Run pre-script (e.g., feature extraction check)
                        if not run_pre_script(idea_id, gpu, cfg):
                            logger.warning(
                                "Pre-script failed for %s, marking FAILED",
                                idea_id)
                            _write_failure(
                                self.results_dir / idea_id,
                                "Pre-script failed")
                            _record_failure(self.failure_counts, idea_id)
                            unclaimed.pop(0)
                            continue
                        logger.info("Launching %s on GPU %d: %s",
                                    idea_id, gpu,
                                    ideas[idea_id]["title"][:50])
                        tp = launch(idea_id, gpu, self.results_dir, cfg)
                        self.active[gpu] = tp
                        unclaimed.pop(0)
                    else:
                        unclaimed.pop(0)
            elif not unclaimed:
                if not self.active:
                    logger.info("All ideas completed or skipped!")
                    if not self.once:
                        logger.info("Waiting for new ideas...")
                else:
                    logger.info("No unclaimed ideas. %d training.",
                                len(self.active))
            else:
                logger.info("%d ideas queued, no free GPUs (%d active)",
                            len(unclaimed), len(self.active))

            # 7. Update report
            completed_rows = update_report(self.results_dir, ideas, cfg)

            # 8. Write status.json
            counts = _count_statuses(ideas, self.results_dir)
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
            )

            # 9. Save state
            save_state(self.results_dir, {
                "iteration": self.iteration,
                "failure_counts": self.failure_counts,
            })

            if self.once:
                if self.active:
                    logger.info("--once mode: waiting for active training...")
                    while self.active:
                        time.sleep(5)
                        check_active(self.active, self.results_dir,
                                     cfg, self.failure_counts)
                    ideas = parse_ideas(cfg["ideas_file"])
                    update_report(self.results_dir, ideas, cfg)
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
                m = json.loads((idea_dir / "metrics.json").read_text())
                st = m.get("status", "UNKNOWN")
            except json.JSONDecodeError:
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

    orze = Orze(gpu_ids, cfg, once=args.once)
    orze.run()


if __name__ == "__main__":
    main()
