# Orze — Operations Reference

## CLI

### Run
```bash
orze -c orze.yaml               # Normal run (auto-detect GPUs)
orze -c orze.yaml --gpus 0,1    # Specific GPUs
orze --once                     # Run one cycle and exit
orze -v                         # Verbose/debug logging
```

### Process Control
```bash
orze --stop                     # Gracefully stop running instance
orze --restart                  # Stop + start
orze --disable                  # Persistently disable (survives restarts)
orze --enable                   # Re-enable after disable
```

### Reporting
```bash
orze --report-only              # Regenerate report.md / leaderboard
orze --admin                    # Launch web UI (port 8787)
orze --check                    # Validate config, files, API keys, GPUs
```

### Agent Roles
```bash
orze --role-only <role_name>    # Run one cycle of a configured role
```

Configured in `orze.yaml` under `roles:`. Supported backends: anthropic, gemini, openai, ollama, custom.

### Project Lifecycle
```bash
orze --init [PATH]              # Scaffold new project (train.py, orze.yaml, ideas.md)
orze --upgrade                  # Self-update from PyPI, restart if running
orze --uninstall                # Full uninstall (keeps results)
```

### Service Management
```bash
orze service install [-c orze.yaml] [--method crontab|systemd]
orze service uninstall
orze service status
orze service logs [-n 50]
```

### Config Overrides
```bash
orze --gpus 0,1,2,3             # Override GPU selection
orze --timeout 7200              # Override max training time
orze --poll 60                   # Override loop sleep interval
orze --ideas-md FILE             # Override ideas file
orze --base-config FILE          # Override base config
orze --results-dir DIR           # Override results directory
orze --train-script FILE         # Override training script
```

## Built-in Agents

| Agent | Purpose |
|---|---|
| `research` | LLM-powered idea generation (multi-backend) |
| `bug_fixer` | Auto-detect stalls, zombies, disk issues, auto-restart |
| `archive_ideas` | Move completed ideas from ideas.md → idea_lake.db |
| `rebuild_lake` | Rebuild SQLite archive from results directories |
| `orze_gc` | Delete checkpoints for non-top experiments |
| `hf_discover` | Query HuggingFace Hub for models |
| `manual_notify` | Trigger status notification manually |

Custom agents: any Python script via `mode: script` in roles config.

## Admin Web UI

`orze --admin` → http://localhost:8787

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/api/status` | Machine-readable status |
| GET | `/api/leaderboard` | Top models |
| GET | `/api/runs` | Active + recent runs (paginated) |
| GET | `/api/run/detail?idea_id=...` | Metrics for one run |
| GET | `/api/run/log?idea_id=...` | Tail training log |
| GET | `/api/queue` | Filterable idea queue |
| GET | `/api/ideas` | All parsed ideas |
| GET | `/api/nodes` | Host heartbeats + GPU details |
| GET | `/api/alerts` | Active alerts |
| GET | `/api/config` | Project config (keys masked) |
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
      on: [new_best, failed]       # per-channel override
```

## Evaluation Pipeline

```yaml
eval_script: evaluate.py
eval_args: ["--idea-id", "{idea_id}", "--gpu", "{gpu}"]
eval_timeout: 3600
eval_output: eval_report.json      # skip if exists

post_scripts:                       # optional post-eval steps
  - name: analysis
    script: analyze.py
    args: ["--idea-id", "{idea_id}"]
    timeout: 1800
    output: analysis_done.json
```

Report columns can read metrics from eval output:
```yaml
report:
  columns:
    - key: auc
      source: "eval_report.json:metrics.auc_roc"
```

## Key Files

| File | Purpose |
|---|---|
| `orze.yaml` | Main config |
| `ideas.md` | Idea queue (append-only) |
| `idea_lake.db` | SQLite archive of completed/failed ideas |
| `results/report.md` | Human-readable leaderboard |
| `results/status.json` | Machine-readable status |
| `results/_leaderboard.json` | Top models metadata |

## Sentinel Files

```bash
touch results/.orze_disabled    # Disable watchdog auto-restart
touch results/.orze_stop_all    # Stop all instances gracefully
touch results/.orze_shutdown    # Temporary shutdown (expires after 120s)
rm results/.orze_disabled       # Re-enable watchdog
```

## orze.yaml Reference

```yaml
# Required
train_script: train.py
ideas_file: ideas.md

# Optional paths
base_config: configs/base.yaml
results_dir: results
python: /path/to/venv/bin/python3

# Execution
timeout: 21600              # max training time (seconds)
poll: 30                    # loop sleep interval
stall_minutes: 30           # kill if no log output
max_idea_failures: 3        # skip after N failures
max_fix_attempts: 2         # auto-fix attempts per failure
min_disk_gb: 50             # pause if disk below threshold
orphan_timeout_hours: 6     # reclaim stale claims
auto_upgrade: true          # auto-upgrade from PyPI

# Roles
roles:
  my_role:
    mode: research          # research | script | claude
    backend: anthropic      # anthropic | gemini | openai | ollama | custom
    model: claude-opus-4-6
    rules_file: RULES.md
    cooldown: 600
    timeout: 600

# GC
gc:
  enabled: true
  checkpoints_dir: checkpoints
  keep_top: 50
  keep_recent: 20
  min_free_gb: 100

# Report
report:
  title: "My Project"
  primary_metric: accuracy
  sort: descending
  columns:
    - {key: "accuracy", label: "Acc", fmt: ".4f"}
  views: []

# Notifications
notifications:
  enabled: true
  on: [completed, failed, new_best]
  channels: []

# Cleanup
cleanup:
  interval: 200
  patterns: ["checkpoint_epoch*.pt", "*.tmp"]
```
