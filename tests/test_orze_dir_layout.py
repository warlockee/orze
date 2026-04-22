"""Tests for .orze/ directory layout + config resolution."""
import os
import tempfile
from pathlib import Path
import pytest

from orze.core.config import load_project_config, orze_path


def test_config_sets_orze_dir_keys_relative(tmp_path):
    """Test that load_project_config sets _orze_dir, _project_root, _env_ORZE_* with relative results_dir."""
    config_file = tmp_path / "orze.yaml"
    config_file.write_text("""
results_dir: orze_results
ideas_file: .orze/ideas.md
idea_lake_db: .orze/idea_lake.db
""")
    
    os.chdir(tmp_path)
    cfg = load_project_config(str(config_file))
    
    assert "_orze_dir" in cfg
    assert "_project_root" in cfg
    assert "_env_ORZE_RESULTS_DIR" in cfg
    assert "_env_ORZE_DIR" in cfg
    
    assert Path(cfg["_orze_dir"]) == tmp_path / ".orze"
    assert Path(cfg["_project_root"]) == tmp_path
    assert Path(cfg["_env_ORZE_RESULTS_DIR"]) == tmp_path / "orze_results"
    assert Path(cfg["_env_ORZE_DIR"]) == tmp_path / ".orze"


def test_config_sets_orze_dir_keys_absolute(tmp_path):
    """Test that load_project_config sets keys correctly with absolute results_dir."""
    results_abs = tmp_path / "my_results"
    results_abs.mkdir()
    
    config_file = tmp_path / "orze.yaml"
    config_file.write_text(f"""
results_dir: {results_abs}
ideas_file: .orze/ideas.md
""")
    
    os.chdir(tmp_path)
    cfg = load_project_config(str(config_file))
    
    assert Path(cfg["_orze_dir"]) == tmp_path / ".orze"
    assert Path(cfg["_project_root"]) == tmp_path
    assert Path(cfg["_env_ORZE_RESULTS_DIR"]) == results_abs


def test_config_default_ideas_file(tmp_path):
    """Test that default ideas_file resolves to {orze_dir}/ideas.md."""
    config_file = tmp_path / "orze.yaml"
    config_file.write_text("results_dir: orze_results\n")
    
    os.chdir(tmp_path)
    cfg = load_project_config(str(config_file))
    
    assert Path(cfg["ideas_file"]) == tmp_path / ".orze" / "ideas.md"


def test_config_default_idea_lake_db(tmp_path):
    """Test that default idea_lake_db resolves to {orze_dir}/idea_lake.db."""
    config_file = tmp_path / "orze.yaml"
    config_file.write_text("results_dir: orze_results\n")
    
    os.chdir(tmp_path)
    cfg = load_project_config(str(config_file))
    
    assert Path(cfg["idea_lake_db"]) == tmp_path / ".orze" / "idea_lake.db"


def test_orze_path_all_kinds(tmp_path):
    """Test orze_path() for every valid kind — each returns correct path and auto-creates parent."""
    cfg = {
        "_orze_dir": str(tmp_path / ".orze"),
        "_env_ORZE_RESULTS_DIR": str(tmp_path / "orze_results"),
    }
    
    kinds_and_expected = [
        ("logs", tmp_path / ".orze" / "logs" / "test.log"),
        ("receipts", tmp_path / ".orze" / "receipts" / "rec.json"),
        ("locks", tmp_path / ".orze" / "locks" / "lock"),
        ("triggers", tmp_path / ".orze" / "triggers" / "_trigger_x"),
        ("mcp", tmp_path / ".orze" / "mcp" / "file.json"),
        ("state", tmp_path / ".orze" / "state" / "state.json"),
        ("heartbeats", tmp_path / ".orze" / "heartbeats" / "hb.json"),
        ("backups", tmp_path / ".orze" / "backups" / "backup.tar"),
        ("feedback", tmp_path / ".orze" / "feedback" / "fb.md"),
        ("tmp", tmp_path / ".orze" / "tmp" / "tmp.txt"),
        ("stray", tmp_path / "orze_results" / "stray" / "file.py"),
        ("rules", tmp_path / ".orze" / "rules" / "RULE.md"),
        ("methods", tmp_path / "orze_results" / "methods" / "m.py"),
        ("knowledge", tmp_path / "orze_results" / "knowledge" / "k.md"),
    ]
    
    for kind, expected in kinds_and_expected:
        # Extract name from expected path
        name = expected.name
        result = orze_path(cfg, kind, name)
        assert result == expected, f"orze_path(cfg, {kind!r}, {name!r}) != {expected}"
        # Parent dir should be created
        assert result.parent.exists(), f"Parent dir not created for {kind}"


def test_orze_path_unknown_kind_raises(tmp_path):
    """Test that unknown kind raises ValueError."""
    cfg = {
        "_orze_dir": str(tmp_path / ".orze"),
        "_env_ORZE_RESULTS_DIR": str(tmp_path / "orze_results"),
    }
    
    with pytest.raises(ValueError, match="Unknown kind"):
        orze_path(cfg, "invalid_kind", "name")
