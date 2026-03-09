"""Install/uninstall orze watchdog service via crontab or systemd --user."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

SERVICE_CONFIG_PATH = Path.home() / ".orze_service.json"
_CRON_TAG = "# orze-watchdog-managed"
_SYSTEMD_DIR = Path.home() / ".config" / "systemd" / "user"


def _detect_method():
    """Detect best available service method. Returns 'systemd' or 'crontab'."""
    try:
        result = subprocess.run(
            ["systemctl", "--user", "is-system-running"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode in (0, 1):  # running or degraded both OK
            return "systemd"
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass

    if shutil.which("crontab"):
        return "crontab"

    return None


def _save_service_config(config_file, results_dir, workdir, python,
                         method, stall_threshold, log_file):
    """Write ~/.orze_service.json."""
    data = {
        "config_file": str(Path(config_file).resolve()),
        "results_dir": str(Path(results_dir).resolve()),
        "workdir": str(Path(workdir).resolve()),
        "python": python,
        "method": method,
        "stall_threshold": stall_threshold,
        "log_file": log_file,
        "installed_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    SERVICE_CONFIG_PATH.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def _install_crontab(svc_cfg):
    """Add watchdog to crontab (idempotent via tag)."""
    python = svc_cfg["python"]
    cron_line = f"* * * * * {python} -m orze.service.watchdog {_CRON_TAG}"

    # Read existing crontab
    try:
        result = subprocess.run(
            ["crontab", "-l"], capture_output=True, text=True, timeout=5,
        )
        existing = result.stdout if result.returncode == 0 else ""
    except (FileNotFoundError, subprocess.TimeoutExpired):
        existing = ""

    # Remove old orze-watchdog lines
    lines = [l for l in existing.splitlines() if _CRON_TAG not in l]
    lines.append(cron_line)

    # Write back
    new_crontab = "\n".join(lines) + "\n"
    proc = subprocess.run(
        ["crontab", "-"], input=new_crontab, text=True,
        capture_output=True, timeout=5,
    )
    if proc.returncode != 0:
        raise RuntimeError(f"crontab install failed: {proc.stderr}")


def _uninstall_crontab():
    """Remove watchdog from crontab."""
    try:
        result = subprocess.run(
            ["crontab", "-l"], capture_output=True, text=True, timeout=5,
        )
        if result.returncode != 0:
            return
        existing = result.stdout
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return

    lines = [l for l in existing.splitlines() if _CRON_TAG not in l]
    new_crontab = "\n".join(lines) + "\n" if lines else ""

    if not lines:
        subprocess.run(["crontab", "-r"], capture_output=True, timeout=5)
    else:
        subprocess.run(
            ["crontab", "-"], input=new_crontab, text=True,
            capture_output=True, timeout=5,
        )


def _install_systemd(svc_cfg):
    """Create systemd --user service + timer."""
    _SYSTEMD_DIR.mkdir(parents=True, exist_ok=True)
    python = svc_cfg["python"]
    workdir = svc_cfg["workdir"]
    log_file = svc_cfg["log_file"]
    config_file = svc_cfg["config_file"]

    # Main orze service
    service_unit = f"""\
[Unit]
Description=Orze GPU experiment orchestrator
After=network.target

[Service]
Type=simple
WorkingDirectory={workdir}
ExecStart={python} -m orze.cli -c {config_file}
Restart=on-failure
RestartSec=30
StandardOutput=append:{log_file}
StandardError=append:{log_file}

[Install]
WantedBy=default.target
"""
    (_SYSTEMD_DIR / "orze.service").write_text(service_unit, encoding="utf-8")

    # Watchdog timer (stall checks every 5 min)
    timer_unit = f"""\
[Unit]
Description=Orze watchdog stall checker

[Timer]
OnBootSec=60
OnUnitActiveSec=300

[Install]
WantedBy=timers.target
"""
    (_SYSTEMD_DIR / "orze-watchdog.timer").write_text(timer_unit, encoding="utf-8")

    # Watchdog oneshot service
    watchdog_unit = f"""\
[Unit]
Description=Orze watchdog stall check

[Service]
Type=oneshot
ExecStart={python} -m orze.service.watchdog
"""
    (_SYSTEMD_DIR / "orze-watchdog.service").write_text(watchdog_unit, encoding="utf-8")

    # Enable lingering so user services survive logout
    subprocess.run(
        ["loginctl", "enable-linger", os.environ.get("USER", "")],
        capture_output=True, timeout=10,
    )

    subprocess.run(["systemctl", "--user", "daemon-reload"],
                   capture_output=True, timeout=10)
    subprocess.run(["systemctl", "--user", "enable", "--now", "orze.service"],
                   capture_output=True, timeout=10)
    subprocess.run(["systemctl", "--user", "enable", "--now", "orze-watchdog.timer"],
                   capture_output=True, timeout=10)


def _uninstall_systemd():
    """Stop, disable, and remove systemd --user units."""
    units = ["orze.service", "orze-watchdog.timer", "orze-watchdog.service"]
    for unit in units:
        subprocess.run(
            ["systemctl", "--user", "stop", unit],
            capture_output=True, timeout=10,
        )
        subprocess.run(
            ["systemctl", "--user", "disable", unit],
            capture_output=True, timeout=10,
        )
        unit_file = _SYSTEMD_DIR / unit
        if unit_file.exists():
            unit_file.unlink()

    subprocess.run(["systemctl", "--user", "daemon-reload"],
                   capture_output=True, timeout=10)


def install(config_file, method="auto", stall_threshold=1800):
    """Install the orze watchdog service.

    Args:
        config_file: Path to orze.yaml
        method: 'auto', 'crontab', or 'systemd'
        stall_threshold: Seconds before heartbeat is considered stale
    """
    from orze.core.config import load_project_config

    # Load config to get results_dir
    cfg = load_project_config(config_file)
    results_dir = cfg["results_dir"]
    workdir = str(Path(config_file).resolve().parent)
    python = sys.executable
    log_file = f"/tmp/orze_{os.environ.get('USER', 'user')}.log"

    if method == "auto":
        method = _detect_method()
        if not method:
            print("\033[31mError: Neither systemd --user nor crontab available.\033[0m")
            sys.exit(1)

    # Save service config
    svc_cfg = _save_service_config(
        config_file, results_dir, workdir, python,
        method, stall_threshold, log_file,
    )

    print(f"\n\033[1mOrze — Service Install\033[0m")
    print(f"  Method         : {method}")
    print(f"  Config         : {svc_cfg['config_file']}")
    print(f"  Results dir    : {svc_cfg['results_dir']}")
    print(f"  Python         : {python}")
    print(f"  Stall threshold: {stall_threshold}s")
    print(f"  Log file       : {log_file}")
    print()

    if method == "crontab":
        _install_crontab(svc_cfg)
        print("\033[32m  Crontab entry installed (checks every 1 minute).\033[0m")
    elif method == "systemd":
        _install_systemd(svc_cfg)
        print("\033[32m  systemd --user service + watchdog timer installed.\033[0m")
    else:
        print(f"\033[31m  Unknown method: {method}\033[0m")
        sys.exit(1)

    print(f"\n  Service config: {SERVICE_CONFIG_PATH}")
    print(f"\n  Check status:  \033[36morze service status\033[0m")
    print(f"  View logs:     \033[36morze service logs\033[0m")
    print(f"  Uninstall:     \033[36morze service uninstall\033[0m")


def uninstall():
    """Uninstall the orze watchdog service (tries both methods for safety)."""
    print(f"\n\033[1mOrze — Service Uninstall\033[0m")

    # Try crontab
    try:
        _uninstall_crontab()
        print("  Crontab entries removed.")
    except Exception as e:
        print(f"  Crontab cleanup: {e}")

    # Try systemd
    try:
        _uninstall_systemd()
        print("  systemd units removed.")
    except Exception as e:
        print(f"  systemd cleanup: {e}")

    # Remove service config
    if SERVICE_CONFIG_PATH.exists():
        SERVICE_CONFIG_PATH.unlink()
        print(f"  Service config removed: {SERVICE_CONFIG_PATH}")

    print("\n\033[32m  Uninstalled. Orze will no longer auto-restart.\033[0m")
