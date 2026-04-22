"""Role process lifecycle management and ideas.md corruption guard.

CALLING SPEC:
    check_active_roles(active_roles: Dict[str, RoleProcess],
                       ideas_file: str = "ideas.md")
        -> list[tuple[str, Outcome]]
        Poll all running role processes. Kill any that exceed their timeout.
        Returns list of (role_name, outcome) for roles that finished this call.
        Outcome is one of OUTCOME_OK / OUTCOME_TIMEOUT / OUTCOME_ERROR /
        OUTCOME_SOFT_FAILURE (see Outcome enum below).
        Releases filesystem locks and checks ideas.md integrity after each role.

    Outcome (IntEnum)
        OUTCOME_OK             = 0  process exited 0 and produced its output
        OUTCOME_TIMEOUT        = 1  killed after exceeding rp.timeout
        OUTCOME_ERROR          = 2  process exited non-zero for a real reason
                                     (script bug, config error, etc.)
        OUTCOME_SOFT_FAILURE   = 3  exit 0 but writes_ideas_file=True role
                                     did not modify ideas.md
        OUTCOME_RATE_LIMITED   = 4  process exited non-zero because the
                                     backing LLM hit a billing / usage /
                                     rate-limit ceiling (Claude CLI
                                     "out of extra usage", Gemini 429,
                                     Anthropic 529, etc.). Transient —
                                     do NOT count toward circuit-breaker
                                     consecutive_failures; just retry on
                                     the next scheduled cycle.

    is_success(outcome) -> bool
        Back-compat helper: True iff outcome == OUTCOME_OK.
"""
import logging
import re
import shutil
import time
from enum import IntEnum
from pathlib import Path
from typing import Dict

from orze.engine.process import RoleProcess, _terminate_and_reap
from orze.core.fs import _fs_unlock

logger = logging.getLogger("orze")


class Outcome(IntEnum):
    OK = 0
    TIMEOUT = 1
    ERROR = 2
    SOFT_FAILURE = 3
    RATE_LIMITED = 4


OUTCOME_OK = Outcome.OK
OUTCOME_TIMEOUT = Outcome.TIMEOUT
OUTCOME_ERROR = Outcome.ERROR
OUTCOME_SOFT_FAILURE = Outcome.SOFT_FAILURE
OUTCOME_RATE_LIMITED = Outcome.RATE_LIMITED


# Signals in role stdout/stderr that indicate an LLM rate-limit /
# billing cap / quota exhaustion, NOT a script bug. Matched against the
# last ~4 KB of the role log. Patterns are lowercased-case-insensitive.
_RATE_LIMIT_SIGNATURES = (
    "out of extra usage",            # Claude CLI (`claude -p`)
    "rate limit exceeded",           # generic
    "429 too many requests",         # HTTP 429
    "resource_exhausted",            # Google API
    "quota exceeded",                # Google / OpenAI
    "insufficient_quota",            # OpenAI
    "overloaded_error",              # Anthropic API
    "429 resource has been exhausted",
)


def _is_role_stalled(rp: "RoleProcess", stall_minutes: int) -> bool:
    """True if rp's log file hasn't grown for ``stall_minutes``.

    Mutates rp's `_last_log_size` and `_stall_since` fields across calls
    so the caller (check_active_roles polling loop) doesn't need to
    manage timer state. Returns False immediately when stall_minutes
    <= 0 (feature disabled).
    """
    if stall_minutes <= 0:
        return False
    now = time.time()
    try:
        current_size = rp.log_path.stat().st_size
    except OSError:
        current_size = rp._last_log_size
    if current_size > rp._last_log_size:
        rp._last_log_size = current_size
        rp._stall_since = 0.0
        return False
    if rp._stall_since == 0.0:
        rp._stall_since = now
        return False
    return (now - rp._stall_since) > stall_minutes * 60


def _is_rate_limit_exit(log_path: "Path") -> bool:
    """Scan the last ~4 KB of a role log for LLM rate-limit signatures.

    Rate-limit hits surface as exit-code-non-zero from the Claude/Gemini
    CLI with the error printed to stdout/stderr. We don't want those
    one-off transient events to advance the consecutive-failure counter
    and trip the circuit-breaker backoff — the role was blocked by the
    provider, not broken.
    """
    try:
        with open(log_path, "rb") as fh:
            try:
                fh.seek(-4096, 2)
            except OSError:
                fh.seek(0)
            tail = fh.read().decode("utf-8", errors="ignore").lower()
    except OSError:
        return False
    return any(sig in tail for sig in _RATE_LIMIT_SIGNATURES)


def is_success(outcome: "Outcome | bool") -> bool:
    """True iff the outcome represents a clean successful cycle."""
    if isinstance(outcome, bool):
        return outcome
    return outcome == OUTCOME_OK


# Track consecutive soft failures per role (exit 0 but no ideas.md modification)
_consecutive_soft_failures: Dict[str, int] = {}
_SOFT_FAILURE_ERROR_THRESHOLD = 5


def check_active_roles(active_roles: Dict[str, "RoleProcess"],
                       ideas_file: str = "ideas.md",
                       role_stall_minutes: int = 0) -> list:
    """Check running role processes. Returns list of (role_name, Outcome) tuples.

    ``role_stall_minutes`` (default 0 = disabled): kill a running role
    whose log file hasn't grown for this many minutes, even when the
    wall-clock timeout hasn't elapsed. Catches claude-CLI hangs that
    produce a 0-byte log and would otherwise burn the full timeout.
    """
    finished = []
    for role_name in list(active_roles.keys()):
        rp = active_roles[role_name]
        ret = rp.process.poll()
        elapsed = time.time() - rp.start_time

        if ret is None:
            # Still running — check wall-clock timeout first, then stall.
            if elapsed > rp.timeout:
                logger.warning("[ROLE TIMEOUT] %s after %.0fm — killing",
                               role_name, elapsed / 60)
                _terminate_and_reap(rp.process, f"role {role_name}")
                rp.close_log()
                _fs_unlock(rp.lock_dir)
                del active_roles[role_name]
                finished.append((role_name, OUTCOME_TIMEOUT))
            elif _is_role_stalled(rp, role_stall_minutes):
                logger.warning("[ROLE STALL] %s — no log output for %dm, "
                               "killing", role_name, role_stall_minutes)
                _terminate_and_reap(rp.process, f"role {role_name}")
                rp.close_log()
                _fs_unlock(rp.lock_dir)
                del active_roles[role_name]
                finished.append((role_name, OUTCOME_TIMEOUT))
            continue

        # Process exited
        rp.close_log()
        _fs_unlock(rp.lock_dir)
        outcome: Outcome
        if ret == 0:
            # Only apply the ideas-modified soft-failure check to roles
            # whose job IS to append to ideas.md (research / research_gemini).
            # Strategy roles (professor, data_analyst, engineer, thinker,
            # code_evolution) modify RESEARCH_RULES.md / *.py / _retrospection.txt
            # instead — their launcher sets rp.writes_ideas_file = False.
            if not getattr(rp, "writes_ideas_file", True):
                logger.info("%s cycle %d completed", role_name, rp.cycle_num)
                _consecutive_soft_failures.pop(role_name, None)
                outcome = OUTCOME_OK
            else:
                ideas_modified = _ideas_were_modified(ideas_file, rp)
                if ideas_modified:
                    logger.info("%s cycle %d completed", role_name, rp.cycle_num)
                    _consecutive_soft_failures.pop(role_name, None)
                    outcome = OUTCOME_OK
                else:
                    count = _consecutive_soft_failures.get(role_name, 0) + 1
                    _consecutive_soft_failures[role_name] = count
                    logger.warning("%s cycle %d exited 0 but ideas.md was not modified "
                                   "(soft failure %d/%d)",
                                   role_name, rp.cycle_num, count,
                                   _SOFT_FAILURE_ERROR_THRESHOLD)
                    if count >= _SOFT_FAILURE_ERROR_THRESHOLD:
                        logger.error("%s has %d consecutive soft failures "
                                     "(exit 0, no output) — role may be misconfigured",
                                     role_name, count)
                    outcome = OUTCOME_SOFT_FAILURE
        else:
            if ret == 42 or _is_rate_limit_exit(rp.log_path):
                logger.warning("%s cycle %d hit LLM rate-limit (exit %d) "
                               "— transient, not counting toward "
                               "consecutive_failures",
                               role_name, rp.cycle_num, ret)
                outcome = OUTCOME_RATE_LIMITED
            else:
                logger.warning("%s cycle %d failed (exit %d), see %s",
                               role_name, rp.cycle_num, ret, rp.log_path)
                outcome = OUTCOME_ERROR

        # CORRUPTION GUARD: check if ideas.md was truncated by the role
        _check_ideas_integrity(ideas_file, rp)

        del active_roles[role_name]
        finished.append((role_name, outcome))

    return finished


def _ideas_were_modified(ideas_file: str, rp: "RoleProcess") -> bool:
    """Check if ideas.md was modified by the role (size or idea count changed).

    Three credit signals, in order:

    1. `ideas_consumed_during_run > 0`: this daemon's consumption phase
       credited the role mid-run — same-daemon case.
    2. ideas.md mtime > `ideas_md_mtime_pre`: ideas.md was touched on
       the shared filesystem during the role's lifetime. Catches the
       cross-daemon case where a DIFFERENT daemon wiped ideas.md (and
       so only bumped its own active_roles' counters).
    3. Post-exit size/count differs from pre-snapshot: fallback for
       roles whose output never got consumed.
    """
    if getattr(rp, "ideas_consumed_during_run", 0) > 0:
        return True
    ideas_path = Path(ideas_file)
    if not ideas_path.exists():
        return False
    pre_mtime = getattr(rp, "ideas_md_mtime_pre", 0.0)
    try:
        current_mtime = ideas_path.stat().st_mtime
        if pre_mtime > 0 and current_mtime > pre_mtime:
            return True
    except OSError:
        pass
    if rp.ideas_pre_size == 0:
        # No pre-snapshot; can't tell — assume modified to avoid false positives
        return True
    try:
        current_size = ideas_path.stat().st_size
        if current_size != rp.ideas_pre_size:
            return True
        # Size same — check idea count
        current_text = ideas_path.read_text(encoding="utf-8")
        current_count = len(re.findall(r"^## idea-[a-z0-9]+:", current_text,
                                       re.MULTILINE))
        return current_count != rp.ideas_pre_count
    except OSError:
        return False


def _prune_corrupt_archive(archive_dir: Path, basename: str,
                           keep: int = 5) -> None:
    """Keep at most ``keep`` most-recent ``<basename>.corrupt.*`` files."""
    try:
        candidates = sorted(archive_dir.glob(f"{basename}.corrupt.*"),
                            key=lambda p: p.stat().st_mtime, reverse=True)
    except OSError:
        return
    for extra in candidates[keep:]:
        try:
            extra.unlink()
        except OSError:
            pass


def cleanup_stale_corrupt_files(ideas_path: Path,
                                cfg: dict = None,
                                archive_dir=None,
                                keep: int = 5) -> int:
    """Move legacy ``ideas.md.corrupt.*`` files into .orze/backups/corrupt/
    and keep only the ``keep`` most-recent. Returns the number moved.
    Idempotent; safe to call on every startup.
    """
    if archive_dir is None and cfg:
        from orze.core.config import orze_path
        archive_dir = orze_path(cfg, "backups") / "corrupt"
    elif archive_dir is None:
        # Fallback for legacy calls without cfg
        archive_dir = ideas_path.parent / "_corrupt_ideas"
    try:
        archive_dir.mkdir(parents=True, exist_ok=True)
    except OSError:
        return 0
    moved = 0
    pattern = f"{ideas_path.name}.corrupt.*"
    for stray in ideas_path.parent.glob(pattern):
        if stray.parent == archive_dir:
            continue
        try:
            shutil.move(str(stray), str(archive_dir / stray.name))
            moved += 1
        except OSError:
            continue
    _prune_corrupt_archive(archive_dir, ideas_path.name, keep=keep)
    return moved


def _read_ingest_state(ideas_path: Path) -> dict:
    """Read the ingest-state sidecar: {size, mtime, ts}."""
    sidecar = ideas_path.with_suffix(ideas_path.suffix + ".ingest_state.json")
    try:
        import json
        return json.loads(sidecar.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}


def mark_ingest(ideas_path: Path) -> None:
    """Record a successful legitimate consumption/wipe of ideas.md.

    The shrink-is-corruption detector uses the recorded timestamp to
    distinguish a DESIGNED post-ingest wipe (legitimate) from a rogue
    role truncating the file (real corruption). Callers: the ideas-
    ingester after successfully moving queued ideas into idea_lake.
    """
    import json
    sidecar = ideas_path.with_suffix(ideas_path.suffix + ".ingest_state.json")
    try:
        st = ideas_path.stat()
        sidecar.write_text(
            json.dumps({"size": st.st_size, "mtime": st.st_mtime,
                        "ts": time.time()}),
            encoding="utf-8",
        )
    except OSError as e:
        logger.debug("mark_ingest failed: %s", e)


# Ingest happened within this many seconds → shrink is expected, not corrupt.
_INGEST_GRACE_WINDOW_S = 120.0


def _check_ideas_integrity(ideas_file: str, rp: "RoleProcess"):
    """Detect and auto-restore ideas.md if a role truncated/corrupted it.

    A shrink is only flagged as corruption when NEITHER of the following
    is true:
      (a) a successful ingest was recorded in the sidecar within the
          last ``_INGEST_GRACE_WINDOW_S`` seconds (designed wipe), OR
      (b) the sidecar's recorded size matches (or is larger than) the
          current size (the file already was small before this role).
    """
    ideas_path = Path(ideas_file)
    if not ideas_path.exists() or rp.ideas_pre_size == 0:
        return

    current_size = ideas_path.stat().st_size
    # Quick check: if file grew or stayed same, it's fine
    if current_size >= rp.ideas_pre_size:
        return

    # Shrink — but was this a legitimate post-ingest wipe?
    ingest_state = _read_ingest_state(ideas_path)
    ingest_ts = ingest_state.get("ts", 0.0)
    if ingest_ts and (time.time() - ingest_ts) < _INGEST_GRACE_WINDOW_S:
        logger.debug(
            "ideas.md shrunk after %s cycle %d but ingest recorded "
            "%.0fs ago — treating as designed wipe, not corruption",
            rp.role_name, rp.cycle_num, time.time() - ingest_ts)
        return
    # Also defend against: sidecar recorded the small post-wipe size; the
    # role was invoked with a pre-snapshot taken BEFORE the wipe but only
    # ran AFTER. Compare against recorded post-ingest size, not pre-snapshot.
    recorded_size = ingest_state.get("size")
    if (isinstance(recorded_size, (int, float))
            and current_size >= recorded_size):
        logger.debug(
            "ideas.md size %d >= last-ingest size %d — no corruption",
            current_size, int(recorded_size))
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
            # Save the corrupted file for forensics — in a dedicated
            # subdir of the results/ tree, keeping only the last 5.
            corrupt_dir = ideas_path.parent / "_corrupt_ideas"
            try:
                corrupt_dir.mkdir(parents=True, exist_ok=True)
            except OSError:
                corrupt_dir = ideas_path.parent
            corrupt_path = corrupt_dir / f"{ideas_path.name}.corrupt.{int(time.time())}"
            shutil.copy2(str(ideas_path), str(corrupt_path))
            _prune_corrupt_archive(corrupt_dir, ideas_path.name, keep=5)
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
