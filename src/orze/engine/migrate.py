"""Versioned layout migrations for the .orze/ dir.

CALLING SPEC:
    CURRENT_LAYOUT -> int
    read_layout_version(orze_dir: Path) -> int
    write_layout_version(orze_dir: Path, version: int) -> None
    migrate_v0_to_v1(project_root, orze_dir, results_dir, dry_run=False) -> list[str]
    _ensure_migrated(project_root, orze_dir, results_dir) -> None  # fast path
"""
import json
import logging
import os
import shutil
import time
from pathlib import Path

logger = logging.getLogger("orze.migrate")

CURRENT_LAYOUT = 1


def read_layout_version(orze_dir: Path) -> int:
    """Read layout version from .orze/state/version.json.
    
    Returns 0 if version file doesn't exist (legacy layout).
    """
    vf = orze_dir / "state" / "version.json"
    if not vf.exists():
        return 0
    try:
        return int(json.loads(vf.read_text())["layout"])
    except Exception:
        return 0


def write_layout_version(orze_dir: Path, version: int) -> None:
    """Write layout version to .orze/state/version.json."""
    vf = orze_dir / "state" / "version.json"
    vf.parent.mkdir(parents=True, exist_ok=True)
    vf.write_text(json.dumps({"layout": version, "updated_at": time.time()}))


def _move(src: Path, dst: Path, dry_run: bool, actions: list[str], rename_on_collision=True):
    """Move src to dst, optionally renaming on collision."""
    if not src.exists():
        return
    if dst.exists() and rename_on_collision:
        dst = dst.with_name(dst.name + f".migrated-{int(time.time())}")
    actions.append(f"{'DRYRUN ' if dry_run else ''}mv {src} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.rename(src, dst)
    except OSError:
        shutil.move(str(src), str(dst))


def migrate_v0_to_v1(project_root: Path, orze_dir: Path, results_dir: Path, dry_run: bool = False) -> list[str]:
    """Relocate legacy files to .orze/ + orze_results/ layout.

    Idempotent: if target exists, skips.
    """
    actions: list[str] = []
    project_root = Path(project_root)
    orze_dir = Path(orze_dir)
    results_dir = Path(results_dir)

    # 1. If legacy `results/` exists and `orze_results/` doesn't, rename.
    legacy_results = project_root / "results"
    if legacy_results.exists() and results_dir.name == "orze_results" and not results_dir.exists() and legacy_results != results_dir:
        _move(legacy_results, results_dir, dry_run, actions)

    rd = results_dir  # canonical name for downstream moves

    # 2. Top-level {ROLE}_RULES.md → .orze/rules/
    for rules_file in project_root.glob("*_RULES.md"):
        _move(rules_file, orze_dir / "rules" / rules_file.name, dry_run, actions)

    # 3. Root ideas.md → .orze/ideas.md (only if target missing)
    root_ideas = project_root / "ideas.md"
    tgt_ideas = orze_dir / "ideas.md"
    if root_ideas.exists() and not tgt_ideas.exists():
        _move(root_ideas, tgt_ideas, dry_run, actions, rename_on_collision=False)

    # 4. Per-role logs + mcp + receipts + triggers from results/
    if rd.exists():
        for child in list(rd.iterdir()):
            name = child.name
            if child.is_dir() and name.startswith("_") and name.endswith("_logs"):
                role = name[1:-len("_logs")]
                _move(child, orze_dir / "logs" / role, dry_run, actions)
            elif child.is_dir() and name.startswith("_") and name.endswith("_lock"):
                role = name[1:-len("_lock")]
                _move(child, orze_dir / "locks" / role, dry_run, actions)
            elif child.is_dir() and name == "_receipts":
                _move(child, orze_dir / "receipts", dry_run, actions)
            elif child.is_file() and name.startswith("_trigger_"):
                _move(child, orze_dir / "triggers" / name, dry_run, actions)
            elif child.is_file() and name.startswith("_host_") and name.endswith(".json"):
                _move(child, orze_dir / "heartbeats" / name[len("_host_"):], dry_run, actions)
            elif child.is_file() and name.endswith("_mcp_config.json") and name.startswith("_"):
                _move(child, orze_dir / "mcp" / name, dry_run, actions)
            elif name == "_bot.log":
                _move(child, orze_dir / "bot.log", dry_run, actions)
            elif name == "idea_lake.db":
                _move(child, orze_dir / "idea_lake.db", dry_run, actions)
            elif name.startswith("ideas.md.safe"):
                ts = int(time.time())
                _move(child, orze_dir / "backups" / f"{name}-{ts}.md", dry_run, actions)
            # state files
            elif name in ("_admin_cache.json", "_config_hashes.json", "_analyst_bridge_state.json", "_bug_fixer_diagnosis.json"):
                _move(child, orze_dir / "state" / name.lstrip("_"), dry_run, actions)
            # feedback
            elif name in ("_failure_feedback.md", "_engineer_report.md"):
                _move(child, orze_dir / "feedback" / name.lstrip("_"), dry_run, actions)
            # knowledge (strip leading _, .txt → .md)
            elif name in ("_analyst_insights.md", "_failure_hypotheses.md", "_error_analysis.md",
                          "_cross_domain_log.md", "_data_audit.md"):
                _move(child, rd / "knowledge" / name.lstrip("_"), dry_run, actions)
            elif name == "_retrospection.txt":
                _move(child, rd / "knowledge" / "retrospection.md", dry_run, actions)
            elif child.is_dir() and name == "_methods":
                _move(child, rd / "methods", dry_run, actions)

    # 5. root ideas.md.corrupt.* and root _corrupt_ideas/ → .orze/backups/corrupt/
    for corrupt in project_root.glob("ideas.md.corrupt.*"):
        ts = int(corrupt.stat().st_mtime) if corrupt.exists() else int(time.time())
        _move(corrupt, orze_dir / "backups" / "corrupt" / f"ideas.md.corrupt-{ts}.md", dry_run, actions)
    legacy_corrupt_dir = project_root / "_corrupt_ideas"
    if legacy_corrupt_dir.exists():
        _move(legacy_corrupt_dir, orze_dir / "backups" / "corrupt_legacy", dry_run, actions)

    # 6. Append .orze/ to .gitignore if missing
    gi = project_root / ".gitignore"
    existing = gi.read_text() if gi.exists() else ""
    if ".orze/" not in existing.splitlines() and ".orze" not in existing.splitlines():
        actions.append(f"{'DRYRUN ' if dry_run else ''}append '.orze/' to {gi}")
        if not dry_run:
            new = existing
            if new and not new.endswith("\n"):
                new += "\n"
            new += ".orze/\n"
            gi.write_text(new)

    return actions


MIGRATIONS = {1: migrate_v0_to_v1}


def _ensure_migrated(project_root, orze_dir, results_dir):
    """Fast path: single stat via version.json. Runs pending migrations otherwise."""
    project_root = Path(project_root)
    orze_dir = Path(orze_dir)
    results_dir = Path(results_dir)
    current = read_layout_version(orze_dir)
    if current >= CURRENT_LAYOUT:
        return
    for v in range(current + 1, CURRENT_LAYOUT + 1):
        fn = MIGRATIONS.get(v)
        if fn is None:
            continue
        try:
            actions = fn(project_root, orze_dir, results_dir, dry_run=False)
            for a in actions:
                logger.info("migrate v%d: %s", v, a)
            write_layout_version(orze_dir, v)
        except Exception as e:
            logger.error("migration v%d failed: %s", v, e)
            raise
