import json
import logging
import os
import shutil
import socket
import time
from pathlib import Path

logger = logging.getLogger("orze")

def _fs_lock(lock_dir: Path, stale_seconds: float = 600) -> bool:
    """Acquire a filesystem lock via atomic mkdir.
    Returns True if acquired, False if held by another.
    Auto-breaks stale locks older than stale_seconds."""
    try:
        lock_dir.mkdir(parents=True, exist_ok=False)
        meta = {"host": socket.gethostname(), "pid": os.getpid(), "time": time.time()}
        (lock_dir / "lock.json").write_text(json.dumps(meta), encoding="utf-8")
        return True
    except FileExistsError:
        # Check for stale lock
        try:
            meta_path = lock_dir / "lock.json"
            if meta_path.exists():
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                if time.time() - meta.get("time", 0) > stale_seconds:
                    logger.warning("Breaking stale lock: %s (age %.0fs)",
                                   lock_dir, time.time() - meta["time"])
                    try:
                        shutil.rmtree(lock_dir)
                    except OSError:
                        return False
                    # Non-recursive retry: attempt mkdir once
                    try:
                        lock_dir.mkdir(parents=True, exist_ok=False)
                        new_meta = {"host": socket.gethostname(), "pid": os.getpid(), "time": time.time()}
                        (lock_dir / "lock.json").write_text(json.dumps(new_meta), encoding="utf-8")
                        return True
                    except FileExistsError:
                        return False
        except Exception:
            pass
        return False

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
def deep_get(obj: dict, dotpath: str, default=None):
    """Get nested dict value by dot path: 'a.b.c' -> obj[a][b][c]."""
    keys = dotpath.split(".")
    for k in keys:
        if isinstance(obj, dict):
            obj = obj.get(k, default)
        else:
            return default
    return obj
def _fs_unlock(lock_dir: Path):
    """Release a filesystem lock."""
    try:
        shutil.rmtree(lock_dir)
    except Exception:
        pass
