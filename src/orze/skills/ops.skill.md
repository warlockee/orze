---
name: ops
---

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

## CLI Quick Reference

```
orze [OPTIONS]

Options:
  -c, --config-file PATH   Path to orze.yaml
  --gpus GPU_IDS           Comma-separated GPU IDs (default: auto-detect)
  --timeout SECONDS        Max training time (default: 3600)
  --poll SECONDS           Loop sleep interval (default: 30)
  --once                   Run one cycle and exit
  --report-only            Only regenerate report.md
  --role-only NAME         Run a single agent role once and exit
  --research-only          Alias for --role-only research
  --ideas-md PATH          Ideas file path
  --base-config PATH       Base config YAML path
  --results-dir PATH       Results directory
  --train-script PATH      Training script
  -v, --verbose            Debug logging
```

CLI args override orze.yaml values.

## Admin API

`orze --admin` -> http://localhost:8787

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
