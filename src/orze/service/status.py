"""Status display and log viewing for orze service."""

import json
import os
import socket
import subprocess
import time
from pathlib import Path

from orze.service.watchdog import (
    SERVICE_CONFIG_PATH, load_service_config,
    _read_pid, _is_pid_alive, _is_heartbeat_stale,
)


def _ok(msg):
    return f"\033[32m[x]\033[0m {msg}"


def _no(msg):
    return f"\033[31m[ ]\033[0m {msg}"


def _warn(msg):
    return f"\033[33m[-]\033[0m {msg}"


def _is_crontab_active():
    """Check if our crontab entry exists."""
    try:
        result = subprocess.run(
            ["crontab", "-l"], capture_output=True, text=True, timeout=5,
        )
        return "orze-watchdog-managed" in result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _is_systemd_active():
    """Check if systemd --user service is active."""
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "orze.service"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() == "active"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def _is_systemd_timer_active():
    """Check if systemd --user watchdog timer is active."""
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-active", "orze-watchdog.timer"],
            capture_output=True, text=True, timeout=5,
        )
        return result.stdout.strip() == "active"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def show_status():
    """Display service status with colored output."""
    hostname = socket.gethostname()

    print(f"\n\033[1mOrze — Service Status\033[0m ({hostname})")
    print("=" * 50)

    # 1. Service config
    svc_cfg = load_service_config()
    if not svc_cfg:
        print(f"\n  {_no('No service config found')}")
        print(f"  Run: \033[36morze service install -c orze.yaml\033[0m")
        return

    method = svc_cfg.get("method", "?")
    installed = svc_cfg.get("installed_at", "?")
    print(f"\n  \033[1mConfig:\033[0m")
    print(f"    Method      : {method}")
    print(f"    Installed   : {installed}")
    print(f"    Config file : {svc_cfg.get('config_file', '?')}")
    print(f"    Results dir : {svc_cfg.get('results_dir', '?')}")
    print(f"    Stall thresh: {svc_cfg.get('stall_threshold', 1800)}s")
    print(f"    Log file    : {svc_cfg.get('log_file', '?')}")

    # 2. Watchdog active
    print(f"\n  \033[1mWatchdog:\033[0m")
    if method == "crontab":
        active = _is_crontab_active()
        print(f"    {_ok('Crontab entry active') if active else _no('Crontab entry missing')}")
    elif method == "systemd":
        svc_active = _is_systemd_active()
        timer_active = _is_systemd_timer_active()
        print(f"    {_ok('orze.service active') if svc_active else _no('orze.service not active')}")
        print(f"    {_ok('orze-watchdog.timer active') if timer_active else _no('orze-watchdog.timer not active')}")

    # 3. Orze process
    results_dir = svc_cfg.get("results_dir", "orze_results")
    print(f"\n  \033[1mOrze process:\033[0m")
    pid = _read_pid(results_dir, hostname)
    if pid:
        alive = _is_pid_alive(pid)
        if alive:
            print(f"    {_ok(f'PID {pid} alive')}")
        else:
            print(f"    {_no(f'PID {pid} not running')}")
    else:
        print(f"    {_warn('No PID file found')}")

    # 4. Heartbeat
    threshold = svc_cfg.get("stall_threshold", 1800)
    print(f"\n  \033[1mHeartbeat:\033[0m")
    stale, age = _is_heartbeat_stale(results_dir, hostname, threshold)
    if age > 0:
        if stale:
            print(f"    {_no(f'Stale ({age:.0f}s ago, threshold {threshold}s)')}")
        else:
            print(f"    {_ok(f'Fresh ({age:.0f}s ago)')}")
    else:
        print(f"    {_warn('No heartbeat found')}")

    # 5. Sentinels
    results_path = Path(results_dir)
    print(f"\n  \033[1mSentinels:\033[0m")
    sentinel_found = False
    for name in [".orze_disabled", ".orze_stop_all", ".orze_shutdown"]:
        path = results_path / name
        if path.exists():
            sentinel_found = True
            try:
                age = time.time() - path.stat().st_mtime
                print(f"    {_warn(f'{name} present ({age:.0f}s old)')}")
            except OSError:
                print(f"    {_warn(f'{name} present')}")
    if not sentinel_found:
        print(f"    {_ok('None active')}")

    # 6. Last watchdog restart
    print(f"\n  \033[1mLast restart:\033[0m")
    marker = results_path / f".orze_watchdog_restart_{hostname}.json"
    if marker.exists():
        try:
            data = json.loads(marker.read_text(encoding="utf-8"))
            print(f"    Reason : {data.get('reason', '?')}")
            print(f"    Time   : {data.get('iso', '?')}")
            print(f"    Prev PID: {data.get('prev_pid', '?')}")
        except (json.JSONDecodeError, OSError):
            print(f"    {_warn('Marker file unreadable')}")
    else:
        print(f"    {_ok('No recent watchdog restarts')}")

    print()


def show_logs(n=50):
    """Tail the watchdog log file."""
    svc_cfg = load_service_config()
    if not svc_cfg:
        print("No service config found. Run 'orze service install' first.")
        return

    log_file = Path(svc_cfg.get("log_file", "/tmp/orze_watchdog.log"))
    if not log_file.exists():
        print(f"Log file not found: {log_file}")
        return

    print(f"\033[1mOrze — Logs\033[0m (last {n} lines from {log_file})")
    print("=" * 50)

    lines = log_file.read_text(encoding="utf-8", errors="replace").splitlines()
    for line in lines[-n:]:
        print(line)
