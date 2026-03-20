"""Shared fixtures for orze tests."""
import os
import shutil
import textwrap
from pathlib import Path

import pytest


@pytest.fixture
def tmp_project(tmp_path):
    """Create a minimal orze project directory and chdir into it."""
    orig_dir = os.getcwd()
    os.chdir(tmp_path)

    # Minimal orze.yaml
    (tmp_path / "orze.yaml").write_text(textwrap.dedent("""\
        train_script: train.py
        ideas_file: ideas.md
        results_dir: results
        python: python3
    """))

    # Minimal ideas.md
    (tmp_path / "ideas.md").write_text(textwrap.dedent("""\
        # Ideas

        ## idea-0001: Baseline
        - **Priority**: high

        ```yaml
        learning_rate: 0.001
        epochs: 5
        ```
    """))

    # configs directory
    (tmp_path / "configs").mkdir()
    (tmp_path / "configs" / "base.yaml").write_text("seed: 42\nnoise: 0.1\n")

    # results directory
    (tmp_path / "results").mkdir()

    yield tmp_path

    os.chdir(orig_dir)


@pytest.fixture
def write_train_script(tmp_project):
    """Write the baseline train.py into the tmp_project."""
    from orze.cli import BASELINE_TRAIN_PY

    (tmp_project / "train.py").write_text(BASELINE_TRAIN_PY.strip() + "\n")
    return tmp_project
