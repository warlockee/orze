"""Multi-machine coordination for Orze.

Calling spec
------------
    from orze.engine.cluster import (
        check_cluster_versions, build_machine_status,
        check_stop_all, check_disabled, kill_orphans,
    )

    heartbeats, bad = check_cluster_versions(results_dir)  # -> (list, set)
    machines = build_machine_status(results_dir)            # -> list[dict]
    should_stop, kill_all = check_stop_all(results_dir)     # -> (bool, bool)
    disabled = check_disabled(results_dir)                  # -> bool
    kill_orphans(results_dir, cfg)                          # side-effect only

All functions are pure (no class state). Side-effects are limited to
filesystem reads and process signals.
"""

import logging
import os
import signal
from pathlib import Path

from orze.reporting.state import _read_all_heartbeats, check_heartbeat_versions

logger = logging.getLogger("orze")


def check_cluster_versions(results_dir: Path) -> tuple:
    """Check version compatibility with other nodes.

    Returns (heartbeats, incompatible_hosts) where incompatible_hosts
    is a set of hostnames with major-version mismatches.
    """
    heartbeats = _read_all_heartbeats(results_dir)
    incompatible_hosts = set(check_heartbeat_versions(heartbeats))
    return heartbeats, incompatible_hosts


def build_machine_status(results_dir: Path) -> list:
    """Build machine status from heartbeats for report notifications.

    Returns list of dicts with keys: host, gpus_busy, gpus_total, utilization.
    """
    heartbeats = _read_all_heartbeats(results_dir)
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


def check_stop_all(results_dir: Path) -> tuple:
    """Check for filesystem-based stop signal (.orze_stop_all).

    Returns (should_stop, kill_all). should_stop is True if the file
    exists; kill_all is True if the file content contains 'kill'.
    """
    stop_file = results_dir / ".orze_stop_all"
    if stop_file.exists():
        try:
            content = stop_file.read_text(encoding="utf-8").strip()
        except OSError:
            content = ""
        kill_all = "kill" in content.lower()
        logger.info("Found .orze_stop_all — shutting down (kill_all=%s)",
                    kill_all)
        return True, kill_all
    return False, False


def check_disabled(results_dir: Path) -> bool:
    """Check for persistent disable flag (.orze_disabled).

    Returns True if the file exists (Orze should not start).
    """
    disabled_file = results_dir / ".orze_disabled"
    if disabled_file.exists():
        msg = disabled_file.read_text(encoding="utf-8").strip()
        logger.error("Orze is DISABLED: %s", msg)
        logger.error("Remove %s to re-enable", disabled_file)
        return True
    return False


def kill_orphans(results_dir: Path, cfg: dict):
    """Kill orphaned train/eval processes from a previous Orze instance.

    Scans /proc for reparented (ppid==1) processes whose cmdline references
    results_dir and matches configured script names, then kills their
    process groups.
    """
    my_pid = os.getpid()
    results_str = str(results_dir)
    patterns = [Path(cfg.get("train_script", "train.py")).name]
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
                    continue
                cmdline = Path(f"/proc/{pid}/cmdline").read_bytes()
                cmdline_str = cmdline.decode("utf-8", errors="replace")
                if (results_str in cmdline_str and
                        any(p in cmdline_str for p in patterns)):
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
