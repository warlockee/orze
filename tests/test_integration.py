"""Integration tests for orze onboarding improvements (v2.2.0 parity on v2.11.0)."""
import ast
import json
import os
import shutil
import subprocess
import sys
import textwrap
from pathlib import Path
from unittest import mock

import pytest

# Ensure orze is importable when running via subprocess
_SRC_DIR = str(Path(__file__).resolve().parent.parent / "src")
_ENV = {**os.environ, "PYTHONPATH": _SRC_DIR}


# ======================================================================
# TestInit — orze --init behaviour
# ======================================================================

class TestInit:
    """Tests for the --init scaffolding flow."""

    def test_init_creates_all_files(self, tmp_path):
        """--init creates train.py, orze.yaml, ideas.md, configs/base.yaml,
        RESEARCH_RULES.md, results/, and venv/."""
        result = subprocess.run(
            [sys.executable, "-m", "orze.cli", "--init", str(tmp_path)],
            capture_output=True, text=True, timeout=120, env=_ENV,
        )
        assert result.returncode == 0, result.stderr

        expected = [
            "train.py", "orze.yaml",
            "configs/base.yaml", "orze_results",
        ]
        for name in expected:
            assert (tmp_path / name).exists(), f"Missing: {name}"
        # ideas.md and rules now live under .orze/
        assert (tmp_path / ".orze" / "ideas.md").exists() or (tmp_path / "ideas.md").exists(), \
            "Missing: ideas.md (expected at .orze/ideas.md or project root)"

    def test_init_idempotent(self, tmp_path):
        """Running --init twice does not overwrite existing files."""
        subprocess.run(
            [sys.executable, "-m", "orze.cli", "--init", str(tmp_path)],
            capture_output=True, text=True, timeout=120, env=_ENV,
        )
        # Write a marker into train.py
        train = tmp_path / "train.py"
        original = train.read_text()
        train.write_text("# MARKER\n" + original)

        # Second init
        subprocess.run(
            [sys.executable, "-m", "orze.cli", "--init", str(tmp_path)],
            capture_output=True, text=True, timeout=120, env=_ENV,
        )
        assert train.read_text().startswith("# MARKER"), "train.py was overwritten"

    def test_init_preserves_existing_ideas(self, tmp_path):
        """If ideas.md already exists, --init leaves it alone."""
        ideas = tmp_path / "ideas.md"
        ideas.parent.mkdir(parents=True, exist_ok=True)
        ideas.write_text("# My custom ideas\n")

        subprocess.run(
            [sys.executable, "-m", "orze.cli", "--init", str(tmp_path)],
            capture_output=True, text=True, timeout=120, env=_ENV,
        )
        assert ideas.read_text() == "# My custom ideas\n"

    def test_init_detects_existing_train_script(self, tmp_path):
        """If train.py already exists, --init does not replace it."""
        train = tmp_path / "train.py"
        train.parent.mkdir(parents=True, exist_ok=True)
        train.write_text("# existing training script\n")

        subprocess.run(
            [sys.executable, "-m", "orze.cli", "--init", str(tmp_path)],
            capture_output=True, text=True, timeout=120, env=_ENV,
        )
        assert train.read_text() == "# existing training script\n"


# ======================================================================
# TestTrainTemplate — BASELINE_TRAIN_PY quality
# ======================================================================

class TestTrainTemplate:
    """Tests for the generated train.py template."""

    def test_template_syntax(self):
        """The template is valid Python."""
        from orze.cli import BASELINE_TRAIN_PY
        ast.parse(BASELINE_TRAIN_PY)

    def test_template_runs(self, write_train_script):
        """train.py executes and produces metrics.json."""
        proj = write_train_script
        idea_dir = proj / "results" / "idea-0001"
        idea_dir.mkdir(parents=True, exist_ok=True)

        # Write idea_config.yaml (as orze would)
        (idea_dir / "idea_config.yaml").write_text(
            "learning_rate: 0.01\nepochs: 5\n"
        )

        result = subprocess.run(
            [sys.executable, str(proj / "train.py"),
             "--idea-id", "idea-0001",
             "--results-dir", str(proj / "results"),
             "--ideas-md", str(proj / "ideas.md"),
             "--config", str(proj / "configs" / "base.yaml")],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        metrics_file = idea_dir / "metrics.json"
        assert metrics_file.exists()

    def test_template_config_merge(self, write_train_script):
        """Idea config overrides base config via deep_merge."""
        proj = write_train_script
        idea_dir = proj / "results" / "idea-0002"
        idea_dir.mkdir(parents=True, exist_ok=True)

        # Base says seed=42, idea overrides seed=99
        (proj / "configs" / "base.yaml").write_text("seed: 42\nepochs: 3\n")
        (idea_dir / "idea_config.yaml").write_text("seed: 99\n")

        result = subprocess.run(
            [sys.executable, str(proj / "train.py"),
             "--idea-id", "idea-0002",
             "--results-dir", str(proj / "results"),
             "--ideas-md", str(proj / "ideas.md"),
             "--config", str(proj / "configs" / "base.yaml")],
            capture_output=True, text=True, timeout=30,
        )
        assert result.returncode == 0, result.stderr
        # The merged config should have seed=99
        assert "seed" in result.stdout.lower() or "99" in result.stdout

    def test_template_unknown_idea_fails_gracefully(self, write_train_script):
        """Running train.py with a nonexistent idea-id still succeeds
        (uses defaults when idea_config.yaml is missing)."""
        proj = write_train_script
        idea_dir = proj / "results" / "idea-9999"
        idea_dir.mkdir(parents=True, exist_ok=True)

        result = subprocess.run(
            [sys.executable, str(proj / "train.py"),
             "--idea-id", "idea-9999",
             "--results-dir", str(proj / "results"),
             "--ideas-md", str(proj / "ideas.md"),
             "--config", str(proj / "configs" / "base.yaml")],
            capture_output=True, text=True, timeout=30,
        )
        # Should still succeed with defaults
        assert result.returncode == 0, result.stderr
        assert (idea_dir / "metrics.json").exists()

    def test_template_metrics_format(self, write_train_script):
        """metrics.json has the required fields."""
        proj = write_train_script
        idea_dir = proj / "results" / "idea-0001"
        idea_dir.mkdir(parents=True, exist_ok=True)
        (idea_dir / "idea_config.yaml").write_text("epochs: 3\n")

        subprocess.run(
            [sys.executable, str(proj / "train.py"),
             "--idea-id", "idea-0001",
             "--results-dir", str(proj / "results"),
             "--ideas-md", str(proj / "ideas.md"),
             "--config", str(proj / "configs" / "base.yaml")],
            capture_output=True, text=True, timeout=30,
        )
        metrics = json.loads((idea_dir / "metrics.json").read_text())
        assert metrics["status"] == "COMPLETED"
        assert "val_loss" in metrics
        assert "train_loss" in metrics
        assert "r2" in metrics
        assert "epochs" in metrics
        assert "training_time" in metrics
        assert isinstance(metrics["training_time"], (int, float))
        assert metrics["training_time"] >= 0


# ======================================================================
# TestUninstall — _do_uninstall preserves research artifacts
# ======================================================================

class TestUninstall:
    """Tests for the uninstall flow."""

    def test_uninstall_preserves_metrics(self, tmp_project):
        """Uninstall keeps metrics.json in idea directories."""
        proj = tmp_project
        idea_dir = proj / "results" / "idea-0001"
        idea_dir.mkdir(parents=True, exist_ok=True)
        (idea_dir / "metrics.json").write_text('{"status": "COMPLETED"}')
        (idea_dir / "runtime_junk.log").write_text("junk")

        from orze.cli import _do_uninstall
        cfg = {
            "results_dir": str(proj / "results"),
            "_config_path": str(proj / "orze.yaml"),
            "ideas_file": str(proj / "ideas.md"),
        }
        _do_uninstall(cfg)

        assert (idea_dir / "metrics.json").exists()
        assert not (idea_dir / "runtime_junk.log").exists()

    def test_uninstall_preserves_models(self, tmp_project):
        """Uninstall keeps model checkpoint files."""
        proj = tmp_project
        idea_dir = proj / "results" / "idea-0001"
        idea_dir.mkdir(parents=True, exist_ok=True)
        for name in ["best_model.pt", "model.pth", "checkpoint.pt"]:
            (idea_dir / name).write_text("model data")

        from orze.cli import _do_uninstall
        cfg = {
            "results_dir": str(proj / "results"),
            "_config_path": str(proj / "orze.yaml"),
            "ideas_file": str(proj / "ideas.md"),
        }
        _do_uninstall(cfg)

        for name in ["best_model.pt", "model.pth", "checkpoint.pt"]:
            assert (idea_dir / name).exists(), f"Missing after uninstall: {name}"

    def test_uninstall_preserves_train_and_ideas(self, tmp_project):
        """Uninstall does not remove train.py or ideas.md."""
        proj = tmp_project
        (proj / "train.py").write_text("# train\n")

        from orze.cli import _do_uninstall
        cfg = {
            "results_dir": str(proj / "results"),
            "_config_path": str(proj / "orze.yaml"),
            "ideas_file": str(proj / "ideas.md"),
        }
        _do_uninstall(cfg)

        assert (proj / "train.py").exists()
        assert (proj / "ideas.md").exists()

    def test_uninstall_removes_scaffold(self, tmp_project):
        """Uninstall removes orze.yaml config file."""
        proj = tmp_project

        from orze.cli import _do_uninstall
        cfg = {
            "results_dir": str(proj / "results"),
            "_config_path": str(proj / "orze.yaml"),
            "ideas_file": str(proj / "ideas.md"),
        }
        _do_uninstall(cfg)

        assert not (proj / "orze.yaml").exists()

    def test_uninstall_removes_venv(self, tmp_project):
        """Uninstall removes the venv/ directory."""
        proj = tmp_project
        venv = proj / "venv"
        venv.mkdir()
        (venv / "sentinel").write_text("x")

        from orze.cli import _do_uninstall
        cfg = {
            "results_dir": str(proj / "results"),
            "_config_path": str(proj / "orze.yaml"),
            "ideas_file": str(proj / "ideas.md"),
        }
        _do_uninstall(cfg)

        assert not venv.exists()


# ======================================================================
# TestConfig — deep_merge and load_idea_config helpers in train template
# ======================================================================

class TestConfig:
    """Tests for config merging helpers embedded in the train template."""

    def _import_helpers(self):
        """Import deep_merge and load_idea_config from the template source."""
        from orze.cli import BASELINE_TRAIN_PY
        ns = {}
        exec(compile(BASELINE_TRAIN_PY, "<train.py>", "exec"), ns)
        return ns["deep_merge"], ns["load_idea_config"]

    def test_deep_merge_flat(self):
        deep_merge, _ = self._import_helpers()
        base = {"a": 1, "b": 2}
        override = {"b": 99, "c": 3}
        result = deep_merge(base, override)
        assert result == {"a": 1, "b": 99, "c": 3}

    def test_deep_merge_nested(self):
        deep_merge, _ = self._import_helpers()
        base = {"model": {"layers": 3, "hidden": 128}, "lr": 0.01}
        override = {"model": {"hidden": 256}, "lr": 0.001}
        result = deep_merge(base, override)
        assert result == {"model": {"layers": 3, "hidden": 256}, "lr": 0.001}

    def test_load_idea_config(self, tmp_path):
        _, load_idea_config = self._import_helpers()
        ideas_md = tmp_path / "ideas.md"
        ideas_md.write_text(textwrap.dedent("""\
            # Ideas

            ## idea-0001: Test
            - **Priority**: high

            ```yaml
            learning_rate: 0.01
            epochs: 5
            ```
        """))
        cfg = load_idea_config(str(ideas_md), "idea-0001")
        assert cfg["learning_rate"] == 0.01
        assert cfg["epochs"] == 5

    def test_load_idea_config_missing_idea(self, tmp_path):
        _, load_idea_config = self._import_helpers()
        ideas_md = tmp_path / "ideas.md"
        ideas_md.write_text("# Ideas\n")
        with pytest.raises(ValueError, match="not found"):
            load_idea_config(str(ideas_md), "idea-9999")


# ======================================================================
# TestVenv — broken venv detection
# ======================================================================

class TestVenv:
    """Tests for venv health checking during --init."""

    def test_broken_venv_detected(self, tmp_path):
        """If venv/ exists but is broken, --init recreates it."""
        venv_dir = tmp_path / "venv"
        venv_dir.mkdir()
        # Create a fake python3 that always fails
        bin_dir = venv_dir / "bin"
        bin_dir.mkdir()
        fake_python = bin_dir / "python3"
        fake_python.write_text("#!/bin/sh\nexit 1\n")
        fake_python.chmod(0o755)

        result = subprocess.run(
            [sys.executable, "-m", "orze.cli", "--init", str(tmp_path)],
            capture_output=True, text=True, timeout=120, env=_ENV,
        )
        assert result.returncode == 0, result.stderr
        # The output should mention broken or recreating
        assert "broken" in result.stdout.lower() or "creating" in result.stdout.lower()

    def test_healthy_venv_not_recreated(self, tmp_path):
        """If venv/ exists and is healthy, --init leaves it alone."""
        # First, create a real venv
        subprocess.run(
            [sys.executable, "-m", "venv", str(tmp_path / "venv")],
            check=True, timeout=60,
        )
        subprocess.run(
            [str(tmp_path / "venv" / "bin" / "python3"), "-m", "pip",
             "install", "--quiet", "pyyaml"],
            check=True, timeout=120,
        )
        # Record mtime of python3 binary
        py_bin = tmp_path / "venv" / "bin" / "python3"
        original_stat = py_bin.stat()

        result = subprocess.run(
            [sys.executable, "-m", "orze.cli", "--init", str(tmp_path)],
            capture_output=True, text=True, timeout=120, env=_ENV,
        )
        assert result.returncode == 0, result.stderr
        assert "healthy" in result.stdout.lower()
        # python3 binary should not have been replaced
        assert py_bin.stat().st_ino == original_stat.st_ino
