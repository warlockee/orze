# Orze — Operations & Extension Guide

Orze is a filesystem-coordinated GPU experiment orchestrator. It runs the loop: **generate ideas → train → evaluate → learn → repeat**.

## Mental Model

```
ideas.md → [claim via mkdir] → train on GPU → metrics.json → leaderboard
     ↑                                                           |
     └──── research agent reads results, appends new ideas ──────┘
```

- **ideas.md**: append-only experiment queue (markdown + YAML config blocks)
- **idea_lake.db**: SQLite archive (hot ideas consumed from ideas.md into DB)
- **results/{idea_id}/metrics.json**: the contract — `{"status": "COMPLETED", ...}`
- **Coordination**: atomic `mkdir` for claims, filesystem locks for roles — works across machines on shared storage (NFS/FSx/EFS)

## Common Operations

### Start / Stop
```bash
orze -c orze.yaml                  # start (auto-detect GPUs)
orze -c orze.yaml --gpus 0,1      # specific GPUs
orze --once                        # one cycle then exit
orze --stop                        # graceful stop (training detaches)
orze --disable                     # persistent disable (survives restarts)
orze --enable                      # re-enable
```

### Monitor
```bash
orze --admin                       # web UI at :8787
orze --report-only                 # regenerate leaderboard
orze --check                       # validate config, API keys, GPUs
cat results/status.json            # machine-readable status
cat results/report.md              # human-readable leaderboard
```

### Run a Single Role
```bash
orze --role-only research          # run research agent once
orze --role-only documenter        # any configured role
```

### Service (auto-restart)
```bash
orze service install -c orze.yaml  # install watchdog (crontab or systemd)
orze service status                # check health
orze service uninstall
```

### Sentinel Files (multi-machine control)
```bash
touch results/.orze_stop_all       # stop all nodes
touch results/.orze_disabled       # persistent disable
rm results/.orze_disabled          # re-enable
```

## Extending Orze

### Add a New Role (LLM Agent)
```yaml
# orze.yaml
roles:
  my_analyzer:
    mode: claude                   # or: script, research
    rules_file: ANALYZER_RULES.md  # prompt for Claude
    model: sonnet
    cooldown: 600                  # seconds between runs
    timeout: 300                   # max execution time
    allowed_tools: "Read,Glob,Grep,Bash"
```

Modes:
- `mode: claude` — spawns `claude -p <rules_file>` as subprocess
- `mode: research` — built-in multi-backend LLM agent (gemini/openai/anthropic/ollama)
- `mode: script` — any Python script: `script: my_agent.py`, `args: [...]`

Template vars in rules_file and args: `{ideas_file}`, `{results_dir}`, `{cycle}`, `{gpu_count}`, `{completed}`, `{queued}`, `{role_name}`.

### Add Pre/Post Scripts
```yaml
pre_script: check_features.py            # runs before training (exit 0 = proceed)
pre_args: ["--idea-id", "{idea_id}", "--gpu", "{gpu}"]

eval_script: evaluate.py                 # runs after successful training
eval_args: ["--idea-id", "{idea_id}", "--gpu", "{gpu}"]
eval_output: eval_report.json            # skip if exists

post_scripts:                            # run after eval
  - name: overlay
    script: generate_overlay.py
    args: ["--idea-id", "{idea_id}"]
    timeout: 1800
    output: overlay_done.json            # skip if exists
```

### Training Script Contract
Input (provided by orze):
```bash
CUDA_VISIBLE_DEVICES=N python train.py \
  --idea-id idea-001 --results-dir results \
  --ideas-md ideas.md --config base.yaml
```

Output (required):
```json
// results/{idea_id}/metrics.json
{"status": "COMPLETED", "test_accuracy": 0.92, "training_time": 142.5}
```

### Ideas Format
```markdown
## idea-001: My Experiment
- **Priority**: high
- **Category**: architecture
- **Parent**: none
- **Hypothesis**: Why this might work.

\```yaml
model:
  type: transformer
training:
  lr: 0.001
\```
```

### Report Columns from Eval Output
```yaml
report:
  columns:
    - key: auc
      source: "eval_report.json:metrics.auc_roc"
```

## Diagnosing Problems

| Symptom | Check | Fix |
|---------|-------|-----|
| No experiments running | `cat results/status.json` — check `free_gpus`, `queue_depth` | Add ideas to ideas.md, check `orze --check` |
| Training stuck | `results/{idea_id}/train_output.log` | Set `stall_minutes` in orze.yaml |
| Research agent not producing ideas | `results/_research_logs/cycle_*.log` | Check `rules_file` exists, API keys set |
| Disk full | `df -h` | Set `gc.enabled: true`, `min_disk_gb`, `cleanup.patterns` |
| Ideas duplicating | Config dedup cache at `results/_config_hashes.json` | Normal — dedup auto-skips duplicates |
| Eval not running | Check `eval_script` exists, `eval_output` not already present | Delete stale eval output to re-run |
| Multi-machine desync | `results/_host_*.json` heartbeats | Check shared filesystem mount, version compat |

## Architecture (source: `src/orze/`)

```
engine/
├── orchestrator.py  (657) — Orze class, __init__, run() recipe
├── phases.py        (542) — main loop phases (OrzePhaseMixin)
├── role_runner.py   (499) — agent role lifecycle (RoleContext)
├── lifecycle.py     (390) — startup, shutdown, PID management
├── reporter.py      (329) — notifications + plateau detection
├── launcher.py      (280) — training subprocess launch
├── evaluator.py     (233) — eval script launch & monitoring
├── health.py        (234) — disk, stall, FS health monitoring
├── upgrade.py       (223) — auto-upgrade pipeline
├── scheduler.py     (208) — idea claiming, GPU scheduling
├── failure.py       (208) — failure tracking, auto-fix
├── process.py       (154) — process tracking dataclasses
├── cluster.py       (140) — multi-machine coordination
├── roles.py         (115) — role process health checks
├── config_dedup.py   (85) — config hash deduplication
└── retrospection.py  (67) — periodic analysis runner

core/
├── config.py     — config loading, validation
├── ideas.py      — ideas.md parsing, sweep expansion
└── fs.py         — atomic filesystem ops, locks

agents/
├── research.py   — multi-backend LLM idea generator
├── bug_fixer.py  — auto-diagnosis watchdog
├── train_idea.py — example training script
└── ...           — archive, GC, HF discover, notify

reporting/
├── leaderboard.py    — report generation
├── state.py          — state persistence, heartbeats
└── notifications.py  — telegram/slack/discord/webhook
```

Each module has a calling spec at the top — read it before reading the implementation.

## Admin API

`orze --admin` → http://localhost:8787

| Method | Endpoint | Returns |
|--------|----------|---------|
| GET | `/api/status` | Machine-readable pipeline status |
| GET | `/api/leaderboard` | Top models ranked by primary metric |
| GET | `/api/runs` | Active + recent runs (paginated) |
| GET | `/api/run/detail?idea_id=...` | Metrics for one run |
| GET | `/api/run/log?idea_id=...` | Tail of training log |
| GET | `/api/queue` | Filterable idea queue |
| GET | `/api/nodes` | Host heartbeats + GPU details |
| GET | `/api/alerts` | Active alerts |
| POST | `/api/ideas` | Add new idea |
| POST | `/api/actions/stop` | Stop all instances |
| POST | `/api/actions/kill` | Kill specific run |

## Notifications

Channels: telegram, slack, discord, webhook.

Events: `completed`, `failed`, `new_best`, `heartbeat`, `milestone`, `disk_warning`, `stall`, `shutdown`, `started`, `role_summary`, `upgrading`, `watchdog_restart`.

```yaml
notifications:
  enabled: true
  on: [completed, failed, new_best]
  channels:
    - type: telegram
      bot_token: "..."
      chat_id: "..."
```

## orze.yaml Quick Reference

```yaml
# Required
train_script: train.py
ideas_file: ideas.md

# Paths
base_config: configs/base.yaml
results_dir: results
python: /path/to/venv/bin/python3

# Execution
timeout: 21600
poll: 30
stall_minutes: 30
max_idea_failures: 3
max_fix_attempts: 2
min_disk_gb: 50
orphan_timeout_hours: 6
auto_upgrade: true

# Roles
roles:
  research:
    mode: claude
    rules_file: RESEARCH_RULES.md
    model: sonnet
    cooldown: 300
    timeout: 600

# GC
gc:
  enabled: true
  checkpoints_dir: checkpoints
  keep_top: 50

# Report
report:
  title: "My Project"
  primary_metric: accuracy
  sort: descending
  columns:
    - {key: "accuracy", label: "Acc", fmt: ".4f"}
```
