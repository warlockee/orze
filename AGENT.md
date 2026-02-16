# Orze — Auto-Research Agent

You are setting up and running **orze**, a GPU experiment orchestrator that automates the full research loop: generate ideas → train → evaluate → learn from results → repeat.

Read `orze/RULES.md` for the complete specification. Below is what you need to do.

## Step 1: Understand the project

Explore the user's codebase to answer:
1. **What is the research task?** (classification, detection, generation, etc.)
2. **What framework?** (PyTorch, TensorFlow, JAX)
3. **Where is the data?**
4. **Is there an existing training script?** If yes, you'll adapt it. If no, you'll write one.
5. **How should results be evaluated?** (accuracy, AUC, loss, custom metric)
6. **What Python environment?** (venv path, conda, system python)
7. **How many GPUs?** Run `nvidia-smi --query-gpu=index --format=csv,noheader` to check.

## Step 2: Create the training script

Orze needs a training script that follows this contract:

**Input** (provided by orze via CLI args):
- `CUDA_VISIBLE_DEVICES` env var — which GPU
- `--idea-id idea-001` — which experiment
- `--results-dir results` — output directory
- `--ideas-md ideas.md` — experiment definitions file
- `--config configs/base.yaml` — base config

**Output** (required):
- `results/{idea_id}/metrics.json` with `{"status": "COMPLETED", ...}` or `{"status": "FAILED", "error": "..."}`

The script must:
1. Parse `--ideas-md` to extract the YAML config for `--idea-id`
2. Load `--config` as base config, merge idea's YAML on top
3. Train on the GPU specified by `CUDA_VISIBLE_DEVICES`
4. Write `metrics.json` when done

If the project already has a training script, wrap or adapt it. If not, write one from scratch.

## Step 3: Create `configs/base.yaml`

Infrastructure defaults only. Model architecture comes from each idea's YAML block.

```yaml
training:
  epochs: 10
  batch_size: 16
data:
  path: /path/to/dataset
```

## Step 4: Write seed ideas in `ideas.md`

Create `ideas.md` with 3-5 baseline experiments. Start simple, then iterate.

```markdown
# Ideas

## idea-001: Simple Baseline
- **Priority**: high
- **Category**: baseline
- **Parent**: none
- **Hypothesis**: Establish a baseline with the simplest viable approach.

\```yaml
model:
  type: simple
training:
  lr: 0.001
  epochs: 10
\```

## idea-002: Larger Model
- **Priority**: medium
- **Category**: architecture
- **Parent**: idea-001
- **Hypothesis**: More capacity may improve performance.

\```yaml
model:
  type: larger
training:
  lr: 0.0003
  epochs: 15
\```
```

IDs must be unique, format `idea-NNN`. Priority controls execution order.

## Step 5: Create `orze.yaml`

Project configuration. Use `orze/orze.yaml.example` as reference.

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
min_disk_gb: 20

# Use Claude CLI as the research agent (generates new ideas automatically)
research:
  mode: claude
  rules_file: RESEARCH_RULES.md
  model: sonnet
  cooldown: 300
  timeout: 600

report:
  title: "My Research"
  primary_metric: test_accuracy
  sort: descending
  columns:
    - {key: "test_accuracy", label: "Accuracy", fmt: ".4f"}
    - {key: "test_loss", label: "Loss", fmt: ".4f"}
    - {key: "training_time", label: "Time(s)", fmt: ".0f"}
```

## Step 6: Create `RESEARCH_RULES.md`

This file is YOUR prompt — it tells Claude (the research agent) how to generate new ideas. Customize it for the project's domain. It must include:

1. **What the research goal is**
2. **What's in the results** — tell Claude to read `{results_dir}/report.md`
3. **What makes a good idea** — domain-specific guidance
4. **The exact idea format** — so ideas parse correctly
5. **Template variables** — use `{ideas_file}`, `{results_dir}`, `{cycle}`, `{completed}`, `{queued}`

See `orze/RULES.md` section "Research Rules Contract" for the full spec.

## Step 7: Smoke test

```bash
# Test one training cycle
python orze/farm.py -c orze.yaml --once --gpus 0

# Check results
cat results/report.md

# Test research agent
python orze/farm.py -c orze.yaml --research-only
```

## Step 8: Launch the full loop

```bash
# All GPUs, continuous — research + train + eval on autopilot
python orze/farm.py -c orze.yaml
```

Monitor progress:
- `results/report.md` — leaderboard (auto-updated)
- `results/status.json` — machine-readable status
- `results/_research_logs/` — research agent logs

## Summary of files you create

```
project/
├── orze/                  # (cloned repo — don't modify)
├── orze.yaml              # project config
├── ideas.md               # experiment definitions (append-only)
├── RESEARCH_RULES.md      # prompt for Claude research agent
├── configs/
│   └── base.yaml          # training defaults
├── train.py               # training script (metrics.json contract)
└── results/               # (auto-created by orze)
    ├── report.md
    ├── status.json
    └── idea-001/
        ├── metrics.json
        └── train_output.log
```
