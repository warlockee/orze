"""Orze watchdog — checks if orze is alive and restarts if needed.

Invokable as: python -m orze.service.watchdog
Designed to be called every minute from crontab or every 5 minutes from systemd timer.
"""

import json
import logging
import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

logger = logging.getLogger("orze.watchdog")

SERVICE_CONFIG_PATH = Path.home() / ".orze_service.json"


def load_service_config(path=None):
    """Read ~/.orze_service.json. Returns dict or None."""
    p = Path(path) if path else SERVICE_CONFIG_PATH
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def _read_pid(results_dir, hostname):
    """Read PID from .orze.pid.{hostname}. Returns int or None."""
    pid_file = Path(results_dir) / f".orze.pid.{hostname}"
    if not pid_file.exists():
        # Fall back to legacy single PID file
        pid_file = Path(results_dir) / ".orze.pid"
        if not pid_file.exists():
            return None
    try:
        return int(pid_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return None


def _is_pid_alive(pid):
    """Check if a process is alive via kill(0)."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _is_heartbeat_stale(results_dir, hostname, threshold):
    """Check if the most recent heartbeat for this host is stale.

    Returns (stale: bool, age_seconds: float) or (False, 0) if no heartbeat found.
    """
    results_dir = Path(results_dir)
    best_epoch = 0
    for hb_path in results_dir.glob(f"_host_{hostname}_*.json"):
        try:
            hb = json.loads(hb_path.read_text(encoding="utf-8"))
            epoch = hb.get("epoch", 0)
            if isinstance(epoch, (int, float)) and epoch > best_epoch:
                best_epoch = epoch
        except (json.JSONDecodeError, OSError):
            continue

    if best_epoch == 0:
        return False, 0

    age = time.time() - best_epoch
    return age > threshold, age


def _should_restart(results_dir):
    """Check sentinel files. Returns (should_skip: bool, reason: str)."""
    results_dir = Path(results_dir)

    disabled = results_dir / ".orze_disabled"
    if disabled.exists():
        return True, "disabled (.orze_disabled exists)"

    stop_all = results_dir / ".orze_stop_all"
    if stop_all.exists():
        return True, "stopped (.orze_stop_all exists)"

    shutdown = results_dir / ".orze_shutdown"
    if shutdown.exists():
        try:
            age = time.time() - shutdown.stat().st_mtime
        except OSError:
            age = 0
        if age < 120:
            return True, f"graceful shutdown {age:.0f}s ago (waiting 120s)"

    return False, ""


def _kill_stale(pid):
    """Kill a stale process. SIGTERM first, SIGKILL after 5s."""
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return

    for _ in range(10):
        time.sleep(0.5)
        if not _is_pid_alive(pid):
            return

    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _is_orze_running():
    """Secondary check: pgrep for orze processes (any launch method)."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", r"orze.*orze\.yaml"],
            capture_output=True, timeout=5,
        )
        # Filter out our own watchdog process
        pids = [p for p in result.stdout.decode().split() if p.strip() and int(p) != os.getpid()]
        return len(pids) > 0
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return False


def _launch_orze(svc_cfg):
    """Launch orze as a detached process."""
    python = svc_cfg["python"]
    config_file = svc_cfg["config_file"]
    workdir = svc_cfg.get("workdir", ".")
    log_file = svc_cfg.get("log_file", "/tmp/orze.log")

    with open(log_file, "a") as lf:
        proc = subprocess.Popen(
            [python, "-m", "orze.cli", "-c", config_file],
            cwd=workdir,
            stdout=lf,
            stderr=lf,
            start_new_session=True,
        )
    return proc.pid


def _write_restart_marker(results_dir, hostname, reason, prev_pid=None):
    """Write a marker file so orchestrator can send notifications on startup."""
    results_dir = Path(results_dir)
    marker = results_dir / f".orze_watchdog_restart_{hostname}.json"
    data = {
        "hostname": hostname,
        "reason": reason,
        "prev_pid": prev_pid,
        "timestamp": time.time(),
        "iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    marker.write_text(json.dumps(data, indent=2), encoding="utf-8")


def check_and_restart(svc_cfg):
    """Main watchdog logic: check alive -> check stale -> check sentinels -> restart."""
    hostname = socket.gethostname()
    results_dir = svc_cfg["results_dir"]
    threshold = svc_cfg.get("stall_threshold", 1800)
    log_file = svc_cfg.get("log_file", "/tmp/orze_watchdog.log")

    def _log(msg):
        line = f"{time.strftime('%Y-%m-%d %H:%M:%S')} [watchdog] {msg}"
        logger.info(msg)
        try:
            with open(log_file, "a") as f:
                f.write(line + "\n")
        except OSError:
            pass

    # Check sentinels first — don't restart if disabled/stopped
    skip, reason = _should_restart(results_dir)
    if skip:
        _log(f"Skipping restart: {reason}")
        return

    pid = _read_pid(results_dir, hostname)

    if pid and _is_pid_alive(pid):
        # Process alive — check for stalls
        stale, age = _is_heartbeat_stale(results_dir, hostname, threshold)
        if stale:
            _log(f"Orze PID {pid} alive but heartbeat stale ({age:.0f}s > {threshold}s). Killing.")
            _kill_stale(pid)
            time.sleep(2)
            _write_restart_marker(results_dir, hostname, f"stale heartbeat ({age:.0f}s)", pid)
        else:
            # All good, nothing to do
            return
    elif pid and not _is_pid_alive(pid):
        _log(f"Orze PID {pid} not alive.")
        _write_restart_marker(results_dir, hostname, "process died", pid)
    else:
        # No PID file — check if somehow running anyway
        if _is_orze_running():
            _log("No PID file but orze process found. Skipping.")
            return
        _write_restart_marker(results_dir, hostname, "no PID file found", None)

    # Double-check: no orze.cli already running (race condition guard)
    if _is_orze_running():
        _log("orze already running (pgrep). Skipping launch.")
        return

    # Re-check sentinels (may have changed during stall kill)
    skip, reason = _should_restart(results_dir)
    if skip:
        _log(f"Skipping restart after kill: {reason}")
        return

    new_pid = _launch_orze(svc_cfg)
    _log(f"Restarted orze (new PID {new_pid})")


def main():
    """Entry point for python -m orze.service.watchdog."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    svc_cfg = load_service_config()
    if not svc_cfg:
        print(f"No service config found at {SERVICE_CONFIG_PATH}", file=sys.stderr)
        print("Run 'orze service install -c orze.yaml' first.", file=sys.stderr)
        sys.exit(1)

    check_and_restart(svc_cfg)


if __name__ == "__main__":
    main()
