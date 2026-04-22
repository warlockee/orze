"""Tests for v0->v1 layout migration."""
import time
from pathlib import Path

from orze.engine.migrate import migrate_v0_to_v1


def test_migrate_v0_to_v1_synthetic_legacy_tree(tmp_path):
    """Test migration with a synthetic legacy tree — all expected files moved."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    results_old = project_root / "results"
    results_old.mkdir()
    
    # Create legacy structure
    (results_old / "_research_logs").mkdir()
    (results_old / "_research_logs" / "file.log").write_text("log content")
    
    (results_old / "_receipts").mkdir()
    (results_old / "_receipts" / "foo.json").write_text("{}")
    
    (results_old / "_host_a.json").write_text("{}")
    (results_old / "_trigger_professor").write_text("")
    (results_old / "_bot.log").write_text("bot log")
    (results_old / "ideas.md.safe").write_text("safe backup")
    (results_old / "idea_lake.db").write_text("db")
    (results_old / "_admin_cache.json").write_text("{}")
    (results_old / "_failure_feedback.md").write_text("feedback")
    (results_old / "_analyst_insights.md").write_text("insights")
    (results_old / "_retrospection.txt").write_text("retro")
    
    (results_old / "_methods").mkdir()
    (results_old / "_methods" / "x.py").write_text("method code")
    
    (project_root / "ideas.md").write_text("root ideas")
    (project_root / "ENGINEER_RULES.md").write_text("rules")
    (project_root / "ideas.md.corrupt.1").write_text("corrupt")
    
    orze_dir = project_root / ".orze"
    results_new = project_root / "orze_results"
    
    # Run migration
    actions = migrate_v0_to_v1(project_root, orze_dir, results_new, dry_run=False)
    
    # Assert all expected targets exist at new paths
    assert results_new.exists(), "orze_results/ not created"
    
    assert (orze_dir / "logs" / "research" / "file.log").exists()
    assert (orze_dir / "receipts" / "foo.json").exists()
    assert (orze_dir / "heartbeats" / "a.json").exists()  # _host_a.json -> a.json
    assert (orze_dir / "triggers" / "_trigger_professor").exists()
    assert (orze_dir / "bot.log").exists()
    
    # Backups have timestamped suffixes
    backups = list((orze_dir / "backups").glob("ideas.md.safe-*.md"))
    assert len(backups) == 1
    
    assert (orze_dir / "idea_lake.db").exists()
    assert (orze_dir / "state" / "admin_cache.json").exists()
    assert (orze_dir / "feedback" / "failure_feedback.md").exists()
    assert (results_new / "knowledge" / "analyst_insights.md").exists()
    assert (results_new / "knowledge" / "retrospection.md").exists()  # .txt -> .md
    assert (results_new / "methods" / "x.py").exists()
    assert (orze_dir / "ideas.md").exists()
    assert (orze_dir / "rules" / "ENGINEER_RULES.md").exists()
    
    corrupt_backups = list((orze_dir / "backups" / "corrupt").glob("ideas.md.corrupt-*.md"))
    assert len(corrupt_backups) == 1


def test_migrate_v0_to_v1_idempotent(tmp_path):
    """Test that second run is no-op — no raises, no duplicates."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    results_old = project_root / "results"
    results_old.mkdir()
    
    # Minimal legacy tree
    (results_old / "ideas.md.safe").write_text("safe")
    (project_root / "ENGINEER_RULES.md").write_text("rules")
    
    orze_dir = project_root / ".orze"
    results_new = project_root / "orze_results"
    
    # First run
    migrate_v0_to_v1(project_root, orze_dir, results_new, dry_run=False)
    
    # Count files after first run
    backup_count_1 = len(list((orze_dir / "backups").rglob("*")))
    rules_count_1 = len(list((orze_dir / "rules").rglob("*")))
    
    # Second run should be no-op
    actions2 = migrate_v0_to_v1(project_root, orze_dir, results_new, dry_run=False)
    
    # Should not raise, should not duplicate backups
    backup_count_2 = len(list((orze_dir / "backups").rglob("*")))
    rules_count_2 = len(list((orze_dir / "rules").rglob("*")))
    
    # Counts should be stable (no duplicates created)
    assert backup_count_2 == backup_count_1
    assert rules_count_2 == rules_count_1


def test_migrate_v0_to_v1_gitignore_once(tmp_path):
    """Test that .gitignore has .orze/ exactly once even on second run."""
    project_root = tmp_path / "project"
    project_root.mkdir()
    results_old = project_root / "results"
    results_old.mkdir()
    
    gitignore = project_root / ".gitignore"
    gitignore.write_text("*.pyc\n__pycache__/\n")
    
    orze_dir = project_root / ".orze"
    results_new = project_root / "orze_results"
    
    # First run
    migrate_v0_to_v1(project_root, orze_dir, results_new, dry_run=False)
    
    content1 = gitignore.read_text()
    assert ".orze/" in content1
    count1 = content1.count(".orze/")
    assert count1 == 1
    
    # Second run
    migrate_v0_to_v1(project_root, orze_dir, results_new, dry_run=False)
    
    content2 = gitignore.read_text()
    count2 = content2.count(".orze/")
    assert count2 == 1, "Second run should not duplicate .orze/ in .gitignore"
