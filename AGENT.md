# Orze — Auto-Research Agent

You are setting up **orze** — a system that automates the full research loop: generate ideas, train on GPUs, evaluate, learn from results, repeat.

Read `orze/RULES.md` for the complete technical specification.

## Core Capabilities (v1.11.1+)
- **1M Scale**: SQLite-backed job queue with indexed O(log N) scheduling.
- **Delta Protocol**: Token-saving communication (agents only output config changes).
- **Split-Brain Protection**: Systematic Mount Integrity checks for multi-node clusters.
- **Self-Healing**: Watchdog process with LLM-based error diagnosis.

## What to do

### 1. Understand the project (explore first, don't ask)

Explore the codebase silently. Find:
- What the project does (read README, docs, existing scripts)
- What framework (PyTorch, JAX, etc.)
- Where the data lives and if features are pre-extracted
- What training scripts exist
- What Python environment (venv, conda, system)
- How many GPUs: run `nvidia-smi --query-gpu=index --format=csv,noheader`

### 2. Determine the research goal

Check if `GOAL.md` exists at the project root. Read or create it to align on the primary metric (e.g., AUC-ROC, Accuracy).

Confirm with the user in **one sentence**:
> "I'll set up auto-research optimizing for [metric] on [dataset] using [architecture]. OK?"

### 3. Create `RESEARCH_RULES.md`

This controls idea generation. **IMPORTANT: Codify the Delta Protocol here.**
Tell the research agent:
1. "Use the **Delta Protocol**: If an idea has a `Parent`, only output the YAML keys that differ from that parent."
2. "Prioritize novel architectures over minor hyperparameter tweaks."
3. "Read `results/report.md` to see current leaderboard and avoid duplicates."

### 4. Create or adapt the training script

Orze calls your script with `--idea-id`, `--results-dir`, `--ideas-md`, and `--config`.

**Systematic Implementation Requirement:**
Your script **must** support the **Recursive Delta Protocol**:
1. Load `base_config` from `--config`.
2. Find `idea_id` in `--ideas-md` or `idea_lake.db`.
3. If the idea has a `Parent`, recursively load the parent's fully resolved config first.
4. Merge the current idea's overrides on top.
5. Write `results/{idea_id}/metrics.json` when done.

### 5. Create `orze.yaml`

```yaml
train_script: train.py
ideas_file: ideas.md
base_config: configs/base.yaml
results_dir: results
python: /path/to/venv/bin/python3

timeout: 3600
poll: 30
stall_minutes: 30
max_idea_failures: 3
max_fix_attempts: 2
min_disk_gb: 50

# Optimized 1M-scale Reporting
report:
  title: "Project Leaderboard"
  primary_metric: test_auc
  sort: descending
  columns:
    - {key: "test_auc", label: "AUC", fmt: ".4f"}
    - {key: "training_time", label: "Time(s)", fmt: ".0f"}

roles:
  research:
    mode: claude
    rules_file: RESEARCH_RULES.md
    model: sonnet
```

### 6. Verify Infrastructure (Mount Integrity)

If working on a multi-node cluster (NFS/EFS/FSx), ensure `idea_lake.db` is present on the shared mount. Orze will automatically detect mount failures and silence inconsistent reports to prevent "split-brain" states.

### 7. Smoke test, then launch

```bash
# Test one cycle manually
python orze/farm.py -c orze.yaml --once --gpus 0

# Launch full system (always start both)
nohup python orze/farm.py -c orze.yaml >> results/farm.log 2>&1 &
nohup python orze/bug_fixer.py -c orze.yaml >> results/bug_fixer.log 2>&1 &
```

## The bug-fixer agent

`orze/bug_fixer.py` is a **lifetime companion**. It auto-restarts the farm, kills stuck jobs, and uses an LLM to diagnose tracebacks. **Always keep it running.**
