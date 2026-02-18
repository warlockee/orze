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

### 3. Create `RESEARCH_RULES.md`

This controls what experiments get generated. Write it based on `GOAL.md` and what you learned from the codebase. Include:

1. Research goal (from GOAL.md)
2. Template vars for current state: `{cycle}`, `{completed}`, `{queued}`, `{results_dir}`, `{ideas_file}`
3. Instructions to read `{results_dir}/report.md` for current results
4. Domain-specific guidance (what approaches work for this task)
5. Concrete directions to explore
6. The exact idea format (from `orze/RULES.md`)
7. Rules: append-only, unique IDs, complete YAML configs

**The user edits this file to change research direction.** Make that clear with a comment at the top.

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

report:
  title: "Research Report"
  primary_metric: test_accuracy     # from GOAL.md
  sort: descending
  columns:
    - {key: "test_accuracy", label: "Accuracy", fmt: ".4f"}
    - {key: "test_loss", label: "Loss", fmt: ".4f"}
    - {key: "training_time", label: "Time(s)", fmt: ".0f"}

# Optional: bug-fixer tuning (defaults shown)
# bug_fixer:
#   check_interval: 60
#   stale_training_min: 45
#   stale_eval_min: 60
#   heartbeat_timeout_min: 5
#   max_fixes_per_hour: 3
#   stale_claim_min: 120
#   min_disk_gb: 50
```

### 8. Smoke test, then launch

```bash
# Test one cycle
python orze/farm.py -c orze.yaml --once --gpus 0

# If it works, launch the full system
# IMPORTANT: Always start both farm.py AND bug_fixer.py together.
# The bug-fixer runs alongside farm.py as a self-healing watchdog.

nohup python orze/farm.py -c orze.yaml >> results/farm.log 2>&1 &
nohup python orze/bug_fixer.py -c orze.yaml >> results/bug_fixer.log 2>&1 &
nohup python orze/bot.py -c orze.yaml >> results/bot.log 2>&1 &  # optional, if Telegram configured
```

Run the smoke test first. If it fails, fix the issue and retry. Once it passes, launch all processes.

## The bug-fixer agent

`orze/bug_fixer.py` is a **lifetime companion process** to `farm.py`. It runs forever alongside the main orchestrator, continuously monitoring for issues and self-healing the system.

### What it does

Runs every 60 seconds (configurable) and checks for:
- **Orze dead**: farm.py process not running → auto-restarts it
- **Orze stalled**: log not updating → spawns Claude to diagnose
- **Zombie processes**: defunct python processes
- **Stuck training/eval**: processes exceeding timeout with no CPU activity → auto-kills
- **Stale claims**: ideas claimed but never completed (abandoned by crashed workers)
- **Low disk space**: below threshold → alerts
- **farm.py errors**: scans log for errors originating in orze code → spawns Claude to fix

### How it fixes bugs

For errors in `farm.py`, the bug-fixer spawns a Claude CLI session that:
1. Reads `farm.py` and the recent log
2. Diagnoses the root cause
3. Applies a minimal patch (if it's an orze bug)
4. Commits locally (does NOT push — human reviews)
5. Reports what it did

**Scope is strictly orze platform only.** It will never modify project scripts, training code, eval scripts, configs, or user files. If an error originates from a user's training script, it reports "not an orze bug" and skips.

### Configuration

Add a `bug_fixer` section to `orze.yaml` to customize (all optional):

```yaml
bug_fixer:
  check_interval: 60          # seconds between health checks
  stale_training_min: 45      # kill training after N idle minutes
  stale_eval_min: 60           # kill eval after N idle minutes
  heartbeat_timeout_min: 5     # orze considered stalled after N minutes
  max_fixes_per_hour: 3        # rate limit on Claude fix sessions
  stale_claim_min: 120         # flag claims older than N minutes
  min_disk_gb: 50              # low disk warning threshold
```

### Always start both

When launching orze, **always start bug_fixer.py alongside farm.py**. They are designed to run as a pair:

```bash
nohup python orze/farm.py -c orze.yaml >> results/farm.log 2>&1 &
nohup python orze/bug_fixer.py -c orze.yaml >> results/bug_fixer.log 2>&1 &
nohup python orze/bot.py -c orze.yaml >> results/bot.log 2>&1 &  # optional
```

If farm.py dies, the bug-fixer will detect it within 60 seconds and restart it. If the bug-fixer itself dies, re-launch it manually. Consider using `supervisord`, `systemd`, or a cron watchdog for both processes in production.

## Files you create

```
project/
├── GOAL.md                # Research target (edit to pivot)
├── RESEARCH_RULES.md      # Idea generation strategy (edit to steer)
├── orze.yaml              # Infrastructure config
├── ideas.md               # Experiments (auto-grows)
├── configs/base.yaml      # Training defaults
├── train.py               # Training script
├── orze/                  # Framework (don't edit)
│   ├── farm.py            # Main orchestrator
│   ├── bug_fixer.py       # Self-healing watchdog
│   └── bot.py             # Telegram bot (optional)
└── results/               # Auto-generated
    ├── farm.log            # Orchestrator log
    ├── bug_fixer.log       # Watchdog log
    ├── bot.log             # Telegram bot log
    └── bug_fixer_issues/   # Issue audit trail
```

**To change research direction:** edit `GOAL.md` and/or `RESEARCH_RULES.md`.
