---
name: core
---

## Ideas Format

Each idea is an H2 header with an ID, title, metadata, and YAML config:

```markdown
## idea-001: My Experiment Name
- **Priority**: high
- **Category**: architecture
- **Parent**: none
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
- **Category**: Free-form label for grouping (e.g., architecture, hyperparameter, augmentation, data, loss, ensemble)
- **Parent**: `none` or `idea-XXX` — tracks which idea inspired this one
- **Hypothesis**: Why you think this idea will work — helps interpret results later

### ID Rules
- Format: `idea-NNN` where NNN is zero-padded (e.g., idea-001, idea-042, idea-1337)
- IDs must be unique within ideas.md
- Higher priority ideas run first; within same priority, lower IDs run first

### Append-Only Rule
ideas.md is **append-only**. Only add new ideas — never edit or delete existing ones. Status is tracked by the filesystem (see Experiment Lifecycle), not in this file.

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
QUEUED → CLAIMED → [PRE-CHECK] → TRAINING → COMPLETED or FAILED → [EVAL] → [POST-SCRIPTS]
```

1. **QUEUED**: Idea exists in ideas.md, no `results/{idea_id}/` directory
2. **CLAIMED**: `results/{idea_id}/` created (atomic mkdir), `claim.json` written
3. **PRE-CHECK**: Optional `pre_script` runs (e.g., verify features exist)
4. **TRAINING**: Subprocess running, writing to `train_output.log`
5. **COMPLETED**: `metrics.json` written with `status: COMPLETED`
6. **FAILED**: `metrics.json` written with `status: FAILED` (by script or orze)
7. **EVAL**: Optional `eval_script` runs, writes eval output file
8. **POST-SCRIPTS**: Optional additional scripts run (overlays, analysis, etc.)

### Failure Causes (auto-detected by orze)
- **Timeout**: Training exceeded `timeout` seconds
- **Stalled**: No log output for `stall_minutes` minutes
- **OOM**: CUDA out of memory detected in log
- **Crash**: Non-zero exit code
- **Pre-script failure**: Pre-training check failed (e.g., missing features)

### Executor LLM Fix (Auto-Retry)
When `max_fix_attempts` > 0 in orze.yaml, failed ideas are automatically sent to an LLM for diagnosis. The LLM reads the error log and idea config, then attempts to fix the project code (scripts, configs, utilities — anything except `orze/` and `ideas.md`). If a fix is applied, the idea is re-launched on the same GPU. Fix attempts are tracked per idea and persisted across restarts.

```yaml
max_fix_attempts: 2          # try up to 2 LLM fixes per failed idea
executor_fix:
  model: sonnet              # LLM model (default: sonnet)
  timeout: 300               # max time per fix attempt (default: 300s)
```

Fix logs are saved to `results/_fix_logs/{idea_id}_attempt{N}.log`.

### Reclaiming Failed Ideas
- Delete the `results/{idea_id}/` directory to allow retry
- Or set `max_idea_failures` in orze.yaml to auto-skip after N failures
- Orphaned claims (no metrics.json after `orphan_timeout_hours`) are auto-cleaned

## Config Merging (Your Training Script's Job)

Orze passes both `--ideas-md` and `--config` (base config) to your training script. **Your script is responsible for merging configs:**

1. Load the base config from `--config` (e.g., `configs/base.yaml`)
2. Parse the idea-specific YAML block from `--ideas-md` using `--idea-id`
3. Merge: idea config overrides base config
4. Train with the merged config

This keeps orze generic — it doesn't need to understand your config schema.

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

## Pre-Script Contract (Optional)

If `pre_script` is configured, it runs before each training launch on the claimed GPU.

**Input**: The command from `pre_args` with `{idea_id}` and `{gpu}` substituted, plus `CUDA_VISIBLE_DEVICES`
**Success**: Exit code 0 — training proceeds
**Failure**: Non-zero exit code — idea marked FAILED, training skipped

Use cases: verify features exist, check disk space, validate configs.

## Evaluation Script Contract (Optional)

If `eval_script` is configured, it runs after each successful training.

**Input**: The command from `eval_args` with `{idea_id}` and `{gpu}` substituted
**Output**: The file named in `eval_output` (default: `eval_report.json`)
**Skip**: If the output file already exists, eval is skipped

## Post-Scripts Contract (Optional)

Additional scripts in `post_scripts` list run after eval. Each entry specifies:
- `script`: path to the script
- `args`: list of args with `{idea_id}` and `{gpu}` substitution
- `timeout`: max time in seconds
- `output`: if this file exists, the script is skipped
- `name`: label for logs

Use cases: overlay generation, model export, additional analysis.
