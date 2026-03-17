"""Auto-upgrade logic for Orze.

Calling spec
------------
    from orze.engine.upgrade import UpgradeManager

    mgr = UpgradeManager(results_dir, cfg)

    mgr.pending                                     # -> Optional[str]
    mgr.check_pypi()                                # rate-limited PyPI poll
    mgr.check_sentinel()                            # react to .orze_upgrade file
    mgr.do_upgrade(kill_and_save_fn, remove_pid_fn) # pip install + restart

    kill_and_save_fn: Callable[[], None]
        Orchestrator callback that SIGTERMs children, waits, saves state,
        and closes the idea lake.  Called right before os.execv.

    remove_pid_fn: Callable[[], None]
        Removes the PID file before exec-restart.

Helper:
    parse_version("1.2.3") -> (1, 2, 3)
"""

import json
import logging
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

from orze import __version__
from orze.core.fs import _fs_lock, _fs_unlock
from orze.reporting.notifications import notify

logger = logging.getLogger("orze")


def parse_version(s: str) -> tuple:
    """Parse a dotted version string into a comparable tuple of ints."""
    try:
        return tuple(int(x) for x in s.split(".")[:3])
    except (ValueError, AttributeError):
        return (0,)


class UpgradeManager:
    """Owns auto-upgrade state and logic, decoupled from orchestrator internals."""

    def __init__(self, results_dir: Path, cfg: dict):
        self._results_dir = results_dir
        self._cfg = cfg
        self._pending_upgrade: Optional[str] = None
        self._last_upgrade_check: float = 0.0

    @property
    def pending(self) -> Optional[str]:
        return self._pending_upgrade

    @pending.setter
    def pending(self, value: Optional[str]):
        self._pending_upgrade = value

    def check_pypi(self):
        """Check PyPI for a newer orze version. Rate-limited. Never raises."""
        au_cfg = self._cfg.get("auto_upgrade")
        if not au_cfg:
            return
        if isinstance(au_cfg, bool):
            interval = 3600
        else:
            interval = int(au_cfg.get("interval", 3600))

        if time.time() - self._last_upgrade_check < interval:
            return
        self._last_upgrade_check = time.time()

        try:
            import urllib.request
            req = urllib.request.Request(
                "https://pypi.org/pypi/orze/json",
                headers={"User-Agent": f"orze/{__version__}"},
            )
            with urllib.request.urlopen(req, timeout=8) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            latest = data["info"]["version"]
        except Exception as exc:
            logger.warning("Auto-upgrade: PyPI check failed: %s", exc)
            return

        if parse_version(latest) > parse_version(__version__):
            if self._pending_upgrade != latest:
                logger.info("Auto-upgrade: v%s available (current v%s)",
                            latest, __version__)
            self._pending_upgrade = latest
        else:
            self._pending_upgrade = None

    def check_sentinel(self):
        """Check if another node wrote .orze_upgrade sentinel.

        If the target version is newer, trigger upgrade immediately.
        """
        sentinel = self._results_dir / ".orze_upgrade"
        if not sentinel.exists():
            return
        try:
            target = sentinel.read_text(encoding="utf-8").strip()
        except Exception:
            return

        if parse_version(target) <= parse_version(__version__):
            try:
                sentinel.unlink(missing_ok=True)
            except OSError:
                pass
            return

        logger.info("Auto-upgrade: sentinel found — another node upgraded to v%s, restarting...", target)
        self._pending_upgrade = target

    def do_upgrade(self, kill_and_save_fn, remove_pid_fn):
        """Install pending upgrade, kill everything, and restart via os.execv.

        Args:
            kill_and_save_fn: callback that kills child processes, saves state,
                              and closes the idea lake.
            remove_pid_fn: callback that removes the PID file.
        """
        target = self._pending_upgrade
        if not target:
            return
        logger.info("Auto-upgrade: installing orze==%s (current v%s)...",
                     target, __version__)

        upgrade_lock = self._results_dir / "_upgrade_lock"
        if not _fs_lock(upgrade_lock, stale_seconds=300):
            logger.info("Auto-upgrade: another process is already upgrading, skipping")
            self._pending_upgrade = None
            return

        # After acquiring lock, check if another process already completed
        # this upgrade (our in-memory __version__ is stale but pip is current)
        try:
            from importlib.metadata import version as _pkg_version
            installed = _pkg_version("orze")
        except Exception:
            installed = __version__

        if parse_version(installed) >= parse_version(target):
            logger.info("Auto-upgrade: v%s already installed, restarting without notification", installed)
            _fs_unlock(upgrade_lock)
            self._pending_upgrade = None
            try:
                (self._results_dir / ".orze_shutdown").write_text(
                    f"auto-upgrade restart to v{installed}", encoding="utf-8")
            except OSError:
                pass
            self._restart_in_place(kill_and_save_fn, remove_pid_fn)
            return

        # Write shutdown sentinel so watchdog won't respawn during restart
        try:
            (self._results_dir / ".orze_shutdown").write_text(
                f"auto-upgrade to v{target}", encoding="utf-8")
        except OSError:
            pass

        try:
            notify("upgrading", {
                "host": socket.gethostname(),
                "from_version": __version__,
                "to_version": target,
                "message": f"Upgrading v{__version__} -> v{target}, restarting",
            }, self._cfg)
        except Exception:
            pass

        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", f"orze=={target}",
             "--quiet"],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            logger.error("Auto-upgrade pip install failed (rc=%d): %s",
                         result.returncode, result.stderr[:500])
            self._pending_upgrade = None
            _fs_unlock(upgrade_lock)
            try:
                (self._results_dir / ".orze_shutdown").unlink(missing_ok=True)
            except OSError:
                pass
            return

        # Signal other nodes sharing this results_dir to restart
        try:
            (self._results_dir / ".orze_upgrade").write_text(
                target, encoding="utf-8")
        except Exception:
            pass

        _fs_unlock(upgrade_lock)
        self._restart_in_place(kill_and_save_fn, remove_pid_fn)

    def _restart_in_place(self, kill_and_save_fn, remove_pid_fn):
        """Kill children, save state, and replace this process via os.execv."""
        logger.info("Auto-upgrade: killing active processes and restarting...")

        kill_and_save_fn()
        remove_pid_fn()

        config_path = str(self._cfg.get("_config_path", "orze.yaml"))
        logger.info("Auto-upgrade: restarting (config: %s)", config_path)
        try:
            os.execv(sys.executable,
                     [sys.executable, "-m", "orze.cli", "-c", config_path])
        except OSError as exc:
            logger.error("Auto-upgrade: os.execv failed: %s — restart manually", exc)
            sys.exit(1)
