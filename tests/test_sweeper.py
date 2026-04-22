"""Tests for stray file sweeper."""
import os
import time
from pathlib import Path

from orze_pro.engine.role_runner import sweep_stray, RoleContext


def test_sweep_stray_moves_new_files(tmp_path):
    """Test that stray sweeper moves new files created by role during cycle."""
    project_root = tmp_path
    orze_dir = project_root / ".orze"
    results_dir = project_root / "orze_results"
    orze_dir.mkdir()
    results_dir.mkdir()
    (orze_dir / "state").mkdir()
    
    cfg = {
        "_project_root": str(project_root),
        "_orze_dir": str(orze_dir),
        "_env_ORZE_RESULTS_DIR": str(results_dir),
        "sweep_stray": True,
    }
    
    ctx = RoleContext(
        cfg=cfg,
        results_dir=results_dir,
        gpu_ids=[],
        active_roles={},
        role_states={},
        failure_counts={},
        fix_counts={},
        iteration=1,
    )
    
    cycle_start_ts = time.time() - 60
    
    # Create files at project root
    evil_py = project_root / "evil.py"
    evil_py.write_text("print('evil')")
    evil_py.touch()  # mtime = now (> cycle_start)
    
    analysis_sh = project_root / "analysis.sh"
    analysis_sh.write_text("#!/bin/bash\necho analysis")
    analysis_sh.touch()
    
    readme = project_root / "README.md"
    readme.write_text("# Project")
    readme.touch()
    
    old_py = project_root / "old.py"
    old_py.write_text("print('old')")
    # Set mtime to before cycle start
    old_mtime = cycle_start_ts - 100
    os.utime(old_py, (old_mtime, old_mtime))
    
    # Run sweeper
    moved = sweep_stray(ctx, "engineer", 1, cycle_start_ts)
    
    # Assert evil.py and analysis.sh moved
    assert str(evil_py) in moved
    assert str(analysis_sh) in moved
    assert len(moved) == 2
    
    # Check destinations
    # evil.py is .py -> methods/
    assert (results_dir / "methods" / "engineer" / "cycle_001" / "evil.py").exists()
    # analysis.sh is .sh -> methods/
    assert (results_dir / "methods" / "engineer" / "cycle_001" / "analysis.sh").exists()
    
    # README.md untouched (allowlist)
    assert readme.exists()
    
    # old.py untouched (mtime older than cycle_start)
    assert old_py.exists()
    
    # Check sweeps.jsonl log
    sweeps_log = orze_dir / "state" / "sweeps.jsonl"
    assert sweeps_log.exists()
    lines = sweeps_log.read_text().strip().split("\n")
    assert len(lines) == 1
    
    import json
    entry = json.loads(lines[0])
    assert entry["role"] == "engineer"
    assert entry["cycle"] == 1
    assert len(entry["moved"]) == 2
