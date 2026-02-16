# Orze Rules for LLM Agents

This document tells you everything you need to operate orze — a filesystem-coordinated GPU experiment orchestrator. Read this before writing ideas or interpreting results.

## Overview

Orze runs experiments by:
1. Reading experiment definitions from `ideas.md` (markdown + YAML)
2. Claiming unclaimed ideas via atomic `mkdir` (filesystem lock)
3. Launching training as subprocesses on free GPUs
4. Monitoring health (stalls, OOM, disk space)
5. Running optional post-training evaluation
6. Generating a leaderboard report and machine-readable status

You control it by editing `ideas.md` and reading `results/`.

## Ideas Format

Each idea is an H2 header with an ID, title, metadata, and YAML config:

```markdown
## idea-001: My Experiment Name
- **Priority**: high
- **Category**: architecture
- **Hypothesis**: Why this might work.

\```yaml
model:
  type: simple_cnn
  channels: [32, 64, 128]
training:
  lr: 0.001
  epochs: 5
\```
```

### Required Fields
- **H2 header**: `## idea-NNN: Title` — the ID must be `idea-` followed by digits
- **YAML block**: Contains the experiment config passed to the training script

### Optional Fields
- **Priority**: `critical` > `high` > `medium` (default) > `low` — controls execution order
- **Category**: Free-form label for grouping (e.g., architecture, hyperparameter, augmentation)
- **Hypothesis**: Why you think this idea will work — helps interpret results later

### ID Rules
- Format: `idea-NNN` where NNN is zero-padded (e.g., idea-001, idea-042, idea-1337)
- IDs must be unique within ideas.md
- Higher priority ideas run first; within same priority, lower IDs run first

## The metrics.json Contract

Your training script **must** write `results/{idea_id}/metrics.json` when done:

```json
{
  "status": "COMPLETED",
  "test_accuracy": 0.9234,
  "test_loss": 0.2451,
  "training_time": 142.5,
  "num_params": 1250000
}
```

### Rules
- `status` must be `"COMPLETED"` or `"FAILED"` — this is the only required field
- If `"FAILED"`, include `"error"` with a description
- Add any metrics you want — they can be displayed in the report via `orze.yaml` config
- The training script receives: `--idea-id`, `--results-dir`, `--ideas-md`, `--config`, plus any `train_extra_args` from orze.yaml

## Experiment Lifecycle

```
QUEUED → CLAIMED → TRAINING → COMPLETED or FAILED → [EVALUATED]
```

1. **QUEUED**: Idea exists in ideas.md, no `results/{idea_id}/` directory
2. **CLAIMED**: `results/{idea_id}/` created (atomic mkdir), `claim.json` written
3. **TRAINING**: Subprocess running, writing to `train_output.log`
4. **COMPLETED**: `metrics.json` written with `status: COMPLETED`
5. **FAILED**: `metrics.json` written with `status: FAILED` (by script or orze)
6. **EVALUATED**: Optional eval script ran, wrote its output file

### Failure Causes (auto-detected by orze)
- **Timeout**: Training exceeded `timeout` seconds
- **Stalled**: No log output for `stall_minutes` minutes
- **OOM**: CUDA out of memory detected in log
- **Crash**: Non-zero exit code

### Reclaiming Failed Ideas
- Delete the `results/{idea_id}/` directory to allow retry
- Or set `max_idea_failures` in orze.yaml to auto-skip after N failures
- Orphaned claims (no metrics.json after `orphan_timeout_hours`) are auto-cleaned

## Reading Results

### results/report.md
Auto-generated leaderboard. Columns are configurable via `orze.yaml`. Sorted by primary metric.

### results/status.json
Machine-readable status, updated every iteration:

```json
{
  "timestamp": "2026-02-16T14:30:00",
  "iteration": 142,
  "active": [{"idea_id": "idea-045", "gpu": 3, "elapsed_min": 12.5}],
  "free_gpus": [0, 1, 2, 4, 5, 6, 7],
  "queue_depth": 87,
  "completed": 55,
  "failed": 3,
  "skipped": 2,
  "disk_free_gb": 1024.5,
  "top_results": [...]
}
```

Use this to monitor progress programmatically.

### Per-Idea Files
Each `results/{idea_id}/` contains:
- `claim.json` — who claimed it, when, on which GPU
- `train_output.log` — stdout/stderr from training
- `metrics.json` — final metrics (the contract)
- `eval_output.log` — eval stdout (if eval configured)
- Other files written by the training/eval scripts

## Configuring for Your Task (orze.yaml)

Create `orze.yaml` in your project root to customize behavior:

```yaml
# Required: paths to your scripts and files
train_script: my_train.py
ideas_file: ideas.md
base_config: configs/base.yaml
results_dir: results
python: /path/to/venv/bin/python3

# Extra args passed to train script after the standard 4 args
train_extra_args:
  - "--data-dir"
  - "/path/to/data"

# Extra environment variables for subprocesses
train_extra_env:
  TORCH_HOME: /path/to/cache

# Timeouts
timeout: 3600           # max training time (seconds)
poll: 30                # loop sleep (seconds)

# Health monitoring (0 = disabled)
stall_minutes: 30       # kill if no log growth
max_idea_failures: 3    # skip after N failures
min_disk_gb: 20         # pause if disk < 20GB free
orphan_timeout_hours: 6 # reclaim stale claims

# Post-training evaluation (optional)
eval_script: my_eval.py
eval_args: ["--idea-id", "{idea_id}", "--gpu", "{gpu}"]
eval_timeout: 3600
eval_output: eval_report.json  # checked for skip-if-exists

# Report configuration
report:
  title: "My Research Report"
  primary_metric: test_accuracy
  sort: descending
  columns:
    - {key: "test_accuracy", label: "Accuracy", fmt: ".4f"}
    - {key: "test_loss", label: "Loss", fmt: ".4f"}
    - {key: "training_time", label: "Time(s)", fmt: ".0f"}
```

### Reading Metrics from Other Files
Columns can read from files other than metrics.json using the `source` field:

```yaml
columns:
  - key: "auc_roc"
    label: "AUC"
    fmt: ".4f"
    source: "eval_report.json:metrics.auc_roc"
```

This reads `results/{idea_id}/eval_report.json` → `metrics` → `auc_roc`.

## Training Script Contract

**Input** (provided by orze):
- `CUDA_VISIBLE_DEVICES` environment variable — which GPU to use
- `--idea-id idea-001` — which idea to train
- `--results-dir results` — where to write output
- `--ideas-md ideas.md` — path to ideas file (read your YAML config from here)
- `--config configs/base.yaml` — path to base config
- Any additional args from `train_extra_args`
- Any additional env vars from `train_extra_env`

**Output** (required from your script):
- `results/{idea_id}/metrics.json` with at minimum `{"status": "COMPLETED"}` or `{"status": "FAILED", "error": "..."}`

**That's it.** Write metrics.json when done. Orze handles everything else.

## Evaluation Script Contract (Optional)

If `eval_script` is configured in orze.yaml, it runs after each successful training.

**Input**: The command from `eval_args` with `{idea_id}` and `{gpu}` substituted
**Output**: The file named in `eval_output` (default: `eval_report.json`)
**Skip**: If the output file already exists, eval is skipped

## Best Practices for Writing Ideas

1. **Start with baselines** — simple models first, then iterate
2. **Vary one thing at a time** — easier to attribute improvements
3. **Use priority** — mark promising directions as `high`, speculative as `low`
4. **Include hypotheses** — helps interpret results and plan next ideas
5. **Check the leaderboard** before generating similar ideas — avoid redundant work
6. **Use categories** — group related ideas for easier analysis
7. **Keep YAML configs complete** — don't rely on implicit defaults

## Multi-Machine Setup

On machines sharing a filesystem (NFS, EFS, FSx):

```bash
# Machine 1
python farm.py -c orze.yaml --gpus 0,1,2,3

# Machine 2
python farm.py -c orze.yaml --gpus 0,1,2,3
```

The atomic mkdir prevents duplicate claims. Each machine's `claim.json` records which host claimed what.

## CLI Quick Reference

```
python farm.py [OPTIONS]

Options:
  -c, --config-file PATH   Path to orze.yaml
  --gpus GPU_IDS           Comma-separated GPU IDs (default: auto-detect)
  --timeout SECONDS        Max training time (default: 3600)
  --poll SECONDS           Loop sleep interval (default: 30)
  --once                   Run one cycle and exit
  --report-only            Only regenerate report.md
  --ideas-md PATH          Ideas file path
  --base-config PATH       Base config YAML path
  --results-dir PATH       Results directory
  --train-script PATH      Training script
  -v, --verbose            Debug logging
```

CLI args override orze.yaml values.
