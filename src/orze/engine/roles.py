import logging
import re
import shutil
import time
from pathlib import Path
from typing import Dict

from orze.engine.process import RoleProcess, _terminate_and_reap
from orze.core.fs import _fs_unlock

logger = logging.getLogger("orze")


def check_active_roles(active_roles: Dict[str, "RoleProcess"],
                       ideas_file: str = "ideas.md") -> list:
    """Check running role processes. Returns list of (role_name, success) tuples."""
    finished = []
    for role_name in list(active_roles.keys()):
        rp = active_roles[role_name]
        ret = rp.process.poll()
        elapsed = time.time() - rp.start_time

        if ret is None:
            # Still running — check timeout
            if elapsed > rp.timeout:
                logger.warning("[ROLE TIMEOUT] %s after %.0fm — killing",
                               role_name, elapsed / 60)
                _terminate_and_reap(rp.process, f"role {role_name}")
                rp.close_log()
                _fs_unlock(rp.lock_dir)
                del active_roles[role_name]
                finished.append((role_name, False))
            continue

        # Process exited
        rp.close_log()
        _fs_unlock(rp.lock_dir)
        if ret == 0:
            logger.info("%s cycle %d completed", role_name, rp.cycle_num)
        else:
            logger.warning("%s cycle %d failed (exit %d), see %s",
                           role_name, rp.cycle_num, ret, rp.log_path)

        # CORRUPTION GUARD: check if ideas.md was truncated by the role
        _check_ideas_integrity(ideas_file, rp)

        del active_roles[role_name]
        finished.append((role_name, ret == 0))

    return finished


def _check_ideas_integrity(ideas_file: str, rp: "RoleProcess"):
    """Detect and auto-restore ideas.md if a role truncated/corrupted it.

    Compares current file size and idea count against pre-role snapshot.
    If file shrunk by >10% or lost ideas, restore from the .safe backup.
    """
    ideas_path = Path(ideas_file)
    if not ideas_path.exists() or rp.ideas_pre_size == 0:
        return

    current_size = ideas_path.stat().st_size
    # Quick check: if file grew or stayed same, it's fine
    if current_size >= rp.ideas_pre_size:
        return

    # File shrunk — check idea count to confirm corruption
    try:
        current_text = ideas_path.read_text(encoding="utf-8")
        current_count = len(re.findall(r"^## idea-[a-z0-9]+:", current_text,
                                       re.MULTILINE))
    except OSError:
        current_count = 0

    if current_count >= rp.ideas_pre_count:
        return  # Count OK despite smaller size (unlikely but possible)

    # CORRUPTION DETECTED
    shrink_pct = (1 - current_size / rp.ideas_pre_size) * 100
    lost_ideas = rp.ideas_pre_count - current_count
    logger.error(
        "IDEAS.MD CORRUPTION DETECTED after %s cycle %d! "
        "File shrunk %.0f%% (%d->%d bytes), lost %d ideas (%d->%d). "
        "Restoring from backup.",
        rp.role_name, rp.cycle_num, shrink_pct,
        rp.ideas_pre_size, current_size,
        lost_ideas, rp.ideas_pre_count, current_count)

    # Restore from .safe backup — validate by idea count, not byte size
    backup_path = ideas_path.with_suffix(".md.safe")
    if backup_path.exists():
        try:
            backup_text = backup_path.read_text(encoding="utf-8")
            backup_count = len(re.findall(r"^## idea-[a-z0-9]+:", backup_text,
                                          re.MULTILINE))
        except OSError:
            backup_count = 0

        if backup_count >= rp.ideas_pre_count:
            # Save the corrupted file for forensics
            corrupt_path = ideas_path.with_suffix(
                f".md.corrupt.{int(time.time())}")
            shutil.copy2(str(ideas_path), str(corrupt_path))
            # Restore
            shutil.copy2(str(backup_path), str(ideas_path))
            logger.info("Restored ideas.md from backup (%d ideas). "
                        "Corrupted version saved to %s",
                        backup_count, corrupt_path)
        else:
            logger.error("Backup also has fewer ideas (%d vs expected %d). "
                         "NOT restoring.",
                         backup_count, rp.ideas_pre_count)
    else:
        logger.error("No backup found at %s — cannot restore!", backup_path)
