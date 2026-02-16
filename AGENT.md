# Orze — Auto-Research Agent

You are setting up and running **orze**, a GPU experiment orchestrator that automates the full research loop: generate ideas, train, evaluate, learn from results, repeat.

## Step 0: Read the research goal

The user's research goal is defined in `GOAL.md` at the project root. Read it now.

If `GOAL.md` does not exist, **ask the user** what they want to research. They should create `GOAL.md` with:

```markdown
# Research Goal

## Task
What are you trying to solve? (e.g., "dashcam collision detection from video")

## Dataset
Where is the data? What format? How much?

## Evaluation
What metric matters? (e.g., "AUC-ROC on held-out test set")

## Constraints
- Hardware (e.g., "8x H100 80GB")
- Time budget (e.g., "run overnight")
- Model size limits
- Framework (PyTorch, etc.)

## Starting points
Any known approaches, existing code, or baselines to build from?
```

**Do not proceed until GOAL.md exists.** Everything else flows from it.

## Step 1: Explore the codebase

Read `GOAL.md`, then explore the project to understand:
1. What framework? (PyTorch, TensorFlow, JAX)
2. Is there an existing training script? If yes, you'll adapt it.
3. What Python environment? (look for venv, conda, requirements.txt)
4. How many GPUs? Run `nvidia-smi --query-gpu=index --format=csv,noheader`

## Step 2: Create `RESEARCH_RULES.md`

This is the **most important file** — it controls what experiments get generated. It is read by the research agent (Claude) every cycle. The user can edit it anytime to change research direction.

Write it based on `GOAL.md`. It must include:

1. **Research goal** — copied/adapted from GOAL.md
2. **Current state** — use template vars: `{cycle}`, `{completed}`, `{queued}`
3. **Where to read results** — `{results_dir}/report.md` and `{results_dir}/status.json`
4. **Domain knowledge** — what approaches work for this task, what to try
5. **What to explore** — concrete research directions
6. **Exact idea format** — so ideas parse correctly (see `orze/RULES.md`)
7. **Where to append ideas** — `{ideas_file}`

The user changes research direction by editing this file. Next research cycle picks it up automatically.

## Step 3: Create the training script

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

If the project already has a training script, wrap or adapt it. If not, write one.

## Step 4: Create `configs/base.yaml`

Infrastructure defaults only. Model architecture comes from each idea's YAML block.

## Step 5: Write seed ideas in `ideas.md`

Create `ideas.md` with 3-5 baseline experiments based on `GOAL.md`. Start simple.

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
```

IDs must be unique, format `idea-NNN`. Priority controls execution order.

## Step 6: Create `orze.yaml`

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

## Step 7: Smoke test

```bash
python orze/farm.py -c orze.yaml --once --gpus 0
cat results/report.md
```

## Step 8: Launch

```bash
python orze/farm.py -c orze.yaml
```

## File summary

```
project/
├── GOAL.md                # YOUR RESEARCH TARGET — edit anytime
├── RESEARCH_RULES.md      # HOW TO GENERATE IDEAS — edit to change direction
├── orze.yaml              # infrastructure config
├── ideas.md               # experiments (append-only)
├── configs/base.yaml      # training defaults
├── train.py               # training script
├── orze/                  # framework (don't edit)
└── results/               # auto-generated
    ├── report.md          # leaderboard
    ├── status.json        # machine-readable status
    └── idea-001/
        └── metrics.json
```

**To change what you're researching:** edit `GOAL.md` and `RESEARCH_RULES.md`.
**To change how experiments run:** edit `orze.yaml`.
**Everything else is automatic.**
