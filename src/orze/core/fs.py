import json
import logging
import os
import shutil
import socket
import time
import uuid
from pathlib import Path

logger = logging.getLogger("orze")

def _fs_lock(lock_dir: Path, stale_seconds: float = 600) -> bool:
    """Acquire a filesystem lock via atomic mkdir.
    Returns True if acquired, False if held by another.
    Auto-breaks stale locks older than stale_seconds using atomic rename
    to avoid TOCTOU races between nodes."""
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
                    # Atomic takeover: rename the stale lock dir to a unique name.
                    # Only one node can succeed at this rename — the loser gets OSError.
                    stale_name = lock_dir.with_name(
                        f"{lock_dir.name}._stale_{uuid.uuid4().hex[:12]}"
                    )
                    try:
                        os.rename(str(lock_dir), str(stale_name))
                    except OSError:
                        # Another node already renamed it — they'll get the lock
                        return False
                    # We won the rename race. Clean up the stale dir.
                    try:
                        shutil.rmtree(stale_name)
                    except OSError:
                        pass
                    # Now acquire the (now-free) lock dir
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
    """Write content atomically via tmp+rename with fsync for Lustre safety."""
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_host = "".join(c if c.isalnum() else "_" for c in socket.gethostname())
    tmp = path.with_name(f"{path.name}.{safe_host}.{os.getpid()}.tmp")
    # Write with explicit fsync before close so other Lustre clients see full content
    fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
    try:
        os.write(fd, content.encode("utf-8"))
        os.fsync(fd)
    finally:
        os.close(fd)
    tmp.replace(path)
    # fsync parent directory so the rename is durable and visible on other nodes
    dir_fd = os.open(str(path.parent), os.O_RDONLY)
    try:
        os.fsync(dir_fd)
    finally:
        os.close(dir_fd)
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
