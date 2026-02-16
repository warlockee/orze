# Orze — Auto-Research Agent

You are setting up **orze** — a system that automates the full research loop: generate ideas, train on GPUs, evaluate, learn from results, repeat.

Read `orze/RULES.md` for the complete technical specification.

## What to do

### 1. Understand the project (explore first, don't ask)

Explore the codebase silently. Find:
- What the project does (read README, docs, existing scripts)
- What framework (PyTorch, JAX, etc.)
- Where the data lives
- What training scripts exist
- What Python environment (venv, conda, system)
- How many GPUs: run `nvidia-smi --query-gpu=index --format=csv,noheader`

### 2. Determine the research goal

Check if `GOAL.md` exists at the project root.

**If it exists:** read it and use it.

**If it doesn't exist:** infer the goal from what you found in step 1, then create `GOAL.md`:

```markdown
# Research Goal

## Task
[Inferred from codebase — e.g., "image classification on CIFAR-10"]

## Dataset
[Found at /path/to/data, N samples, format X]

## Evaluation
[Primary metric — e.g., "test accuracy" or "AUC-ROC"]

## Constraints
[GPUs found, disk space, framework]
```

Then confirm with the user in **one sentence**:
> "This looks like a [task] project. I'll set up auto-research optimizing for [metric] on [dataset]. OK?"

If they say yes (or anything not a correction), proceed. If they correct you, update GOAL.md and proceed.

### 3. Create agent rules files

Create these three rules files based on `GOAL.md` and what you learned from the codebase.

#### `RESEARCH_RULES.md` — idea generation

This controls what experiments get generated. Include:

1. Research goal (from GOAL.md)
2. Template vars for current state: `{cycle}`, `{completed}`, `{queued}`, `{results_dir}`, `{ideas_file}`
3. Instructions to read `{results_dir}/report.md` for current results
4. Domain-specific guidance (what approaches work for this task)
5. Concrete directions to explore
6. The exact idea format (from `orze/RULES.md`)
7. Rules: append-only, unique IDs, complete YAML configs

**The user edits this file to change research direction.** Make that clear with a comment at the top.

#### `MONITOR_RULES.md` — health monitoring & alerts

This agent runs frequently and watches over the pipeline. Include instructions to:

1. Read `{results_dir}/status.json` for current pipeline state
2. Read `{results_dir}/report.md` for latest results
3. Check for anomalies: sudden accuracy drops, repeated failures, stalled GPUs, disk pressure
4. Check training logs for warnings (NaN losses, gradient explosions, OOM near-misses)
5. Write a brief status summary to `{results_dir}/monitor.md` with:
   - Current pipeline health (healthy / warning / critical)
   - Active training jobs and their progress
   - Recent completions and their results
   - Any issues detected
6. If critical issues are found, write details to `{results_dir}/alerts.md`

#### `DOCUMENTER_RULES.md` — research documentation

This agent maintains a living research log. Include instructions to:

1. Read `{results_dir}/report.md` for current leaderboard
2. Read `{results_dir}/monitor.md` for pipeline health
3. Read recent experiment configs and results from `{results_dir}/`
4. Maintain `{results_dir}/research_log.md` — a chronological narrative of:
   - What experiments have been tried and why
   - Key findings (what worked, what didn't, and why)
   - Current best approaches and their characteristics
   - Promising directions for future work
5. Keep it concise and insight-dense — this is for humans to read

### 4. Create or adapt the training script

Orze calls your training script with:
```
CUDA_VISIBLE_DEVICES=N python train.py --idea-id idea-001 --results-dir results --ideas-md ideas.md --config configs/base.yaml
```

It must write `results/{idea_id}/metrics.json` with `{"status": "COMPLETED", ...}` when done.

If a training script already exists, **wrap it** — don't rewrite from scratch. Create a thin adapter that:
1. Parses `--ideas-md` to get the YAML config for `--idea-id`
2. Loads `--config` as base, merges idea config on top
3. Calls the existing training code
4. Writes metrics.json

If no training script exists, write one from scratch.

### 5. Create `configs/base.yaml`

Infrastructure defaults only. Model config comes from each idea's YAML block.

### 6. Write seed ideas in `ideas.md`

3-5 baseline experiments. Start simple — the research agent will generate more.

### 7. Create `orze.yaml`

```yaml
train_script: train.py          # your training script from step 4
ideas_file: ideas.md
base_config: configs/base.yaml
results_dir: results
python: /path/to/venv/bin/python3   # from step 1

timeout: 3600
poll: 30
stall_minutes: 30
max_idea_failures: 3
min_disk_gb: 20

roles:
  research:
    mode: claude
    rules_file: RESEARCH_RULES.md
    model: sonnet
    cooldown: 300
    timeout: 600

  monitor:
    mode: claude
    rules_file: MONITOR_RULES.md
    model: haiku
    cooldown: 120
    timeout: 120

  documenter:
    mode: claude
    rules_file: DOCUMENTER_RULES.md
    model: haiku
    cooldown: 900
    timeout: 300

report:
  title: "Research Report"
  primary_metric: test_accuracy     # from GOAL.md
  sort: descending
  columns:
    - {key: "test_accuracy", label: "Accuracy", fmt: ".4f"}
    - {key: "test_loss", label: "Loss", fmt: ".4f"}
    - {key: "training_time", label: "Time(s)", fmt: ".0f"}
```

### 8. Smoke test, then launch

```bash
# Test one cycle
python orze/farm.py -c orze.yaml --once --gpus 0

# If it works, launch the full loop
python orze/farm.py -c orze.yaml
```

Run the smoke test first. If it fails, fix the issue and retry. Once it passes, launch the full loop.

## Files you create

```
project/
├── GOAL.md                # Research target (edit to pivot)
├── RESEARCH_RULES.md      # Idea generation strategy (edit to steer)
├── MONITOR_RULES.md       # Health monitoring agent rules
├── DOCUMENTER_RULES.md    # Documentation agent rules
├── orze.yaml              # Infrastructure config
├── ideas.md               # Experiments (auto-grows)
├── configs/base.yaml      # Training defaults
├── train.py               # Training script
├── orze/                  # Framework (don't edit)
└── results/               # Auto-generated
```

**To change research direction:** edit `GOAL.md` and/or `RESEARCH_RULES.md`.
**To adjust monitoring:** edit `MONITOR_RULES.md`.
**To change documentation style:** edit `DOCUMENTER_RULES.md`.
