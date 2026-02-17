# orze

Auto-research on autopilot. One script, one config, all GPUs.

Orze runs the full research loop: **generate ideas → train → evaluate → learn → repeat**. It coordinates GPUs via filesystem locks (`mkdir`), works across machines, and supports any LLM as the research agent — Claude, GPT, Gemini, local models, or your own script. No databases, no message queues — just files.

**Website:** [orze.ai](https://orze.ai)

## Install

```bash
curl -sL https://raw.githubusercontent.com/warlockee/orze/main/setup.sh | bash
```

This downloads 4 files into `orze/` under your project root. Then tell [**claude**|**gemini**|**codex**|whatever]:

```
do @orze/AGENT.md
```

Your LLM will explore your codebase, create the config, write seed ideas, and launch the loop. That's it.

## How It Works

```
┌──────────────────────────────────────────────────┐
│                    farm.py                        │
│                                                   │
│   ┌─────────┐     ┌─────────┐     ┌──────────┐  │
│   │ Research │────>│  Train  │────>│ Evaluate │  │
│   │ (any LLM)│     │ (GPUs)  │     │          │  │
│   └────▲────┘     └─────────┘     └──────────┘  │
│        │                                │         │
│        └────────── results/ ◄───────────┘         │
│                                                   │
│   ideas.md ◄── research ── report.md              │
└──────────────────────────────────────────────────┘
```

The loop:
1. **Research** — any LLM (Claude, GPT, Gemini, local) reads results, generates new experiment ideas
2. **Parse** `ideas.md` for experiment definitions
3. **Claim** unclaimed ideas via atomic `mkdir`
4. **Launch** training as subprocesses across free GPUs
5. **Monitor** health — stalls, OOM, timeouts, disk space
6. **Evaluate** — run post-training eval scripts (non-blocking)
7. **Notify** — push updates to Telegram, Slack, Discord
8. **Report** — update `results/report.md` leaderboard + `status.json`
9. Sleep, repeat

## Manual Setup (without Claude Code)

```bash
# 1. Install orze
curl -sL https://raw.githubusercontent.com/warlockee/orze/main/setup.sh | bash

# 2. Create your config (see orze/orze.yaml.example)
cp orze/orze.yaml.example orze.yaml
# Edit orze.yaml — set train_script, python path, report columns

# 3. Write seed ideas in ideas.md
# See orze/RULES.md for the exact format

# 4. Run
python orze/farm.py -c orze.yaml
```

## Key Features

- **Agent roles** — multiple agents (research, documenter, analyzer) run alongside training, each with their own cooldown and state
- **LLM-agnostic** — built-in Claude Code support (`mode: claude`), or bring any LLM via `mode: script` (GPT, Gemini, local models, custom pipelines)
- **Multi-GPU** — claims and trains across all GPUs in parallel
- **Parallel eval** — eval scripts launch non-blocking so GPUs stay busy
- **Health monitoring** — stall detection, OOM detection, disk space checks
- **Push notifications** — Telegram, Slack, Discord, or any webhook. Every message includes the top 10 leaderboard
- **Configurable report** — custom columns, metrics from any JSON file
- **Multi-machine** — works across machines on shared filesystems (NFS/EFS/FSx)
- **Failure handling** — auto-skip after N failures, orphan cleanup
- **Atomic coordination** — `mkdir` as lock, no race conditions

## Notifications

Get pinged on your phone when experiments complete, fail, or set new bests.

```yaml
# In orze.yaml
notifications:
  enabled: true
  channels:
    - type: telegram
      bot_token: "your-bot-token"
      chat_id: "your-chat-id"
    - type: slack
      webhook_url: "https://hooks.slack.com/services/..."
    - type: discord
      webhook_url: "https://discord.com/api/webhooks/..."
```

Events: `completed`, `failed`, `new_best`, `report`. Supports per-channel filtering (`on: [new_best, failed]`) and generic webhooks.

## The Contract

Your training script receives these args from orze:

```
CUDA_VISIBLE_DEVICES=N python train.py --idea-id idea-001 --results-dir results --ideas-md ideas.md --config base.yaml
```

Your script must write `results/{idea_id}/metrics.json`:

```json
{"status": "COMPLETED", "test_accuracy": 0.92, "training_time": 142.5}
```

That's it. See [RULES.md](RULES.md) for the full specification.

## Documentation

| File | For |
|------|-----|
| [AGENT.md](AGENT.md) | Claude Code bootstrap — `@orze/AGENT.md` |
| [RULES.md](RULES.md) | Complete LLM-readable specification |
| [orze.yaml.example](orze.yaml.example) | All configuration options |

## CLI Reference

```
python orze/farm.py [OPTIONS]

  -c, --config-file PATH    Path to orze.yaml
  --gpus GPU_IDS            Comma-separated GPU IDs (default: all)
  --once                    Run one cycle and exit
  --report-only             Only regenerate report.md
  --role-only NAME          Run a single agent role once and exit
  --research-only           Alias for --role-only research
  --timeout SECONDS         Override training timeout
  --poll SECONDS            Override loop interval
  --ideas-md PATH           Override ideas file path
  --base-config PATH        Override base config path
  --results-dir PATH        Override results directory
  --train-script PATH       Override training script
  -v, --verbose             Debug logging
```

## License

Apache 2.0
