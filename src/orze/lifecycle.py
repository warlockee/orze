"""Orze lifecycle management — stop, start, restart.

Provides the core logic for `orze stop`, `orze start`, and `orze restart`
subcommands.  Replaces the external shell scripts (shutdown.sh, start.sh,
restart.sh) with built-in equivalents that work on the local node and
signal remote nodes via shared-filesystem sentinels.
"""

import os
import signal
import socket
import subprocess
import sys
import time
from pathlib import Path

# Pattern used to find the orze orchestrator via pgrep/pkill.
# The [o] trick prevents the grep/pgrep process from matching itself.
#
# Match ONLY the detached orchestrator — `python -m orze.cli -c …orze.yaml`
# (daemon mode) and the foreground equivalent after os.execv. A looser
# pattern like `orze.*orze\.yaml` also matches the *caller* shell whose
# cmdline is literally `orze start -c orze.yaml`, producing a phantom
# "already running" error on the very first `orze start` from bash.
_ORZE_PAT = r"[o]rze\.cli.*orze\.yaml"

# Child process script names to kill on stop.
_CHILD_PAT = (
    r"(train_idea|evaluate_dataset|evaluate_idea"
    r"|extract_features|research_agent|validate_idea)[.]py"
)


def _log(prefix, msg):
    print(f"[{prefix}] {msg}", flush=True)


def _read_pid(results_dir: Path, hostname: str):
    """Read PID from .orze.pid.{hostname} or legacy .orze.pid."""
    for name in [f".orze.pid.{hostname}", ".orze.pid"]:
        pid_file = results_dir / name
        if pid_file.exists():
            try:
                return int(pid_file.read_text(encoding="utf-8").strip())
            except (ValueError, OSError):
                pass
    return None


def _is_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


def _kill_pid(pid: int, timeout: int = 10):
    """SIGTERM a process (group), wait, then SIGKILL if needed."""
    if not _is_alive(pid):
        return
    # Try process group first, fall back to single process
    try:
        os.killpg(os.getpgid(pid), signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            return

    for _ in range(timeout * 2):
        if not _is_alive(pid):
            return
        time.sleep(0.5)

    # Force kill
    try:
        os.killpg(os.getpgid(pid), signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            os.kill(pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass


def _pgrep(pattern: str) -> list:
    """Return PIDs matching a pgrep -f pattern (excluding ourselves)."""
    try:
        result = subprocess.run(
            ["pgrep", "-f", pattern],
            capture_output=True, text=True, timeout=5,
        )
        return [
            int(p) for p in result.stdout.strip().split()
            if p.strip() and int(p) != os.getpid()
        ]
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        return []


def _cleanup_gpu_orphans(workdir: str):
    """Kill orphaned processes from our workdir still holding GPUs."""
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return

    for line in result.stdout.strip().split("\n"):
        pid_str = line.strip()
        if not pid_str:
            continue
        try:
            pid = int(pid_str)
        except ValueError:
            continue
        try:
            cmdline = Path(f"/proc/{pid}/cmdline").read_bytes().decode(
                "utf-8", errors="replace")
            if workdir in cmdline:
                _log("stop", f"Releasing GPU from orphaned process {pid}")
                os.kill(pid, signal.SIGTERM)
        except (OSError, ProcessLookupError, PermissionError):
            pass


def _kill_children_of(parent_pid: int, timeout: int = 60):
    """Kill child processes of a specific PID (scoped, not global pkill)."""
    import re
    try:
        result = subprocess.run(
            ["ps", "--ppid", str(parent_pid), "-o", "pid=", "--no-headers"],
            capture_output=True, text=True, timeout=5,
        )
        child_pids = [
            int(p) for p in result.stdout.strip().split() if p.strip()
        ]
    except (FileNotFoundError, subprocess.TimeoutExpired, ValueError):
        child_pids = []

    if not child_pids:
        _log("stop", "No child processes")
        return

    # Filter to only known child script patterns
    filtered = []
    child_re = re.compile(_CHILD_PAT)
    for cpid in child_pids:
        try:
            cmdline = Path(f"/proc/{cpid}/cmdline").read_bytes().decode(
                "utf-8", errors="replace")
            if child_re.search(cmdline):
                filtered.append(cpid)
        except OSError:
            pass

    if not filtered:
        _log("stop", "No matching child processes")
        return

    _log("stop", f"SIGTERM {len(filtered)} child process(es) of PID {parent_pid}")
    for cpid in filtered:
        try:
            os.kill(cpid, signal.SIGTERM)
        except (ProcessLookupError, PermissionError):
            pass

    elapsed = 0
    while elapsed < timeout:
        still_alive = [p for p in filtered if _is_alive(p)]
        if not still_alive:
            _log("stop", f"All child processes exited after {elapsed}s")
            return
        _log("stop",
             f"Waiting for {len(still_alive)} child process(es)... "
             f"({elapsed}/{timeout}s)")
        time.sleep(5)
        elapsed += 5

    still_alive = [p for p in filtered if _is_alive(p)]
    if still_alive:
        _log("stop", f"Timeout — SIGKILL {len(still_alive)} remaining children")
        for cpid in still_alive:
            try:
                os.kill(cpid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass


# ── stop ─────────────────────────────────────────────────────────────

def do_stop(cfg: dict, timeout: int = 60):
    """Stop orze on the local node.

    1. Write .orze_disabled (prevents watchdog restart)
    2. Write .orze_stop_all with "kill" (signals remote nodes)
    3. Kill local orchestrator via PID file
    4. Kill child processes (train, eval, etc.)
    5. Clean up GPU orphans
    """
    results_dir = Path(cfg["results_dir"])
    hostname = socket.gethostname()
    workdir = os.getcwd()

    _log("stop", f"{time.strftime('%c')} — Stopping orze (timeout={timeout}s)")

    # 1. Disable watchdog
    disabled_path = results_dir / ".orze_disabled"
    if results_dir.exists():
        disabled_path.write_text(
            f"Stopped at {time.strftime('%Y-%m-%dT%H:%M:%S')}",
            encoding="utf-8",
        )
        _log("stop", f"Watchdog disabled via {disabled_path}")

    # 2. Signal remote nodes via shared filesystem
    if results_dir.exists():
        (results_dir / ".orze_stop_all").write_text("kill", encoding="utf-8")

    # 3. Kill local orchestrator
    pid = _read_pid(results_dir, hostname)
    if pid and _is_alive(pid):
        _log("stop", f"SIGTERM orchestrator (PID {pid})")
        _kill_pid(pid, timeout=10)
        if _is_alive(pid):
            _log("stop", f"Orchestrator didn't exit, SIGKILL")
            try:
                os.kill(pid, signal.SIGKILL)
            except (ProcessLookupError, PermissionError):
                pass
        _log("stop", "Orchestrator stopped")
    elif pid:
        _log("stop", f"PID file points at PID {pid} but it is not running")

    # Also catch any orze process for OUR config not tracked by PID file
    # (e.g. PID file missing / stale, or a parallel foreground daemon).
    # Scope the pgrep to processes whose command line contains our results_dir
    # or config path so we don't kill other tenants' instances.
    our_results = str(results_dir.resolve())
    config_path = cfg.get("_config_path") or ""
    untracked_killed = 0
    for extra_pid in _pgrep(_ORZE_PAT):
        try:
            cmdline = Path(f"/proc/{extra_pid}/cmdline").read_bytes().decode(
                "utf-8", errors="replace")
            if our_results not in cmdline and config_path not in cmdline:
                _log("stop", f"Skipping orze PID {extra_pid} (belongs to another instance)")
                continue
        except OSError:
            continue
        _log("stop", f"Killing untracked orze process (PID {extra_pid})")
        _kill_pid(extra_pid, timeout=10)
        untracked_killed += 1

    # Summarise — especially helpful when there was no PID file but we still
    # found (or didn't find) an untracked orchestrator via pgrep.
    if pid is None and untracked_killed == 0:
        _log("stop", "Nothing to stop — no PID file, no matching process")
    elif pid is None and untracked_killed > 0:
        _log("stop", f"No PID file but found and killed {untracked_killed} "
             f"untracked orze process(es) for this config")

    # 4. Kill child processes — only children of our orchestrator PID, not globally.
    if pid:
        _kill_children_of(pid, timeout)
    else:
        _log("stop", "No orchestrator PID — skipping child cleanup")

    # 5. GPU orphan cleanup
    _cleanup_gpu_orphans(workdir)

    _log("stop", f"{time.strftime('%c')} — Stop complete")


# ── start ────────────────────────────────────────────────────────────

def do_start(cfg: dict, foreground: bool = False, config_path: str = None,
             gpus: str = None, timeout: int = None):
    """Start orze on the local node.

    1. Check not already running
    2. Clear sentinels (.orze_disabled, .orze_stop_all, .orze_shutdown)
    3. Launch orze (detached daemon or foreground via os.execv)

    Args:
        gpus: Comma-separated GPU IDs (e.g. "0,1,3"). None = auto-detect.
        timeout: Max training time per job in seconds. None = use config.

    Returns PID in daemon mode. In foreground mode, replaces the process
    via os.execv (never returns).
    """
    results_dir = Path(cfg["results_dir"])
    hostname = socket.gethostname()
    config_path = config_path or cfg.get("_config_path", "orze.yaml")
    python = sys.executable
    log_file = str(Path(cfg.get("results_dir", "orze_results")) / "orze.log")
    Path(log_file).parent.mkdir(parents=True, exist_ok=True)

    # 1. Check not already running
    pid = _read_pid(results_dir, hostname)
    if pid and _is_alive(pid):
        _log("start", f"Orze is already running (PID {pid}). "
             f"Use 'orze restart' instead.")
        sys.exit(1)

    # Also check via pgrep
    running = _pgrep(_ORZE_PAT)
    if running:
        _log("start", f"Orze is already running (PID {running[0]}). "
             f"Use 'orze restart' instead.")
        sys.exit(1)

    # 2. Clear sentinels
    results_dir.mkdir(parents=True, exist_ok=True)
    for name in [".orze_disabled", ".orze_stop_all", ".orze_shutdown"]:
        sentinel = results_dir / name
        if sentinel.exists():
            sentinel.unlink(missing_ok=True)
            _log("start", f"Removed {name}")

    # 3. Build command
    cmd = [python, "-m", "orze.cli", "-c", config_path]
    if gpus:
        cmd.extend(["--gpus", gpus])
    if timeout is not None:
        cmd.extend(["--timeout", str(timeout)])

    # 4. Launch
    if foreground:
        gpu_msg = f" on GPUs {gpus}" if gpus else ""
        _log("start", f"Starting orze in foreground{gpu_msg}...")
        os.execv(python, cmd)
        # Never returns

    with open(log_file, "a") as lf:
        proc = subprocess.Popen(
            cmd,
            stdout=lf, stderr=lf,
            stdin=subprocess.DEVNULL,
            start_new_session=True,
        )

    time.sleep(3)
    if proc.poll() is not None:
        _log("start", f"ERROR: Failed to start (exit code {proc.returncode}). "
             f"Check {log_file}")
        sys.exit(1)

    gpu_msg = f" on GPUs {gpus}" if gpus else ""
    _log("start", f"Orze started{gpu_msg} (PID {proc.pid})")
    _log("start", f"Log: {log_file}")
    return proc.pid


# ── restart ──────────────────────────────────────────────────────────

def do_restart(cfg: dict, timeout: int = 60, foreground: bool = False,
               config_path: str = None, gpus: str = None):
    """Restart orze: stop then start."""
    _log("restart", f"{time.strftime('%c')} — Restarting orze")
    do_stop(cfg, timeout=timeout)
    result = do_start(cfg, foreground=foreground, config_path=config_path,
                      gpus=gpus)
    _log("restart", f"{time.strftime('%c')} — Restart complete")
    return result
