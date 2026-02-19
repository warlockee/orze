# orze

Auto-research on autopilot. One script, one config, all GPUs.

Orze runs the full research loop: **generate ideas → train → evaluate → learn → repeat**. It coordinates GPUs via filesystem locks (`mkdir`), works across machines, and supports any LLM as the research agent — Claude, GPT, Gemini, local models, or your own script. No databases, no message queues — just files.

**Website:** [orze.ai](https://orze.ai)

## Install

```bash
curl -sL https://raw.githubusercontent.com/warlockee/orze/main/setup.sh | bash
```

This downloads the orze files into `orze/` under your project root. Then tell [**claude**|**gemini**|**codex**|whatever]:

```
do @orze/AGENT.md
```

Your LLM will explore your codebase, create the config, write seed ideas, and launch the loop. That's it.

## How It Works

```
┌─────────────────────────────────────────────────────┐
│                      farm.py                        │
│                                                     │
│   ┌──────────┐     ┌─────────┐     ┌──────────┐   │
│   │ Research  │────>│  Train  │────>│ Evaluate │   │
│   │ (any LLM) │     │ (GPUs)  │     │          │   │
│   └─────▲────┘     └─────────┘     └──────────┘   │
│         │                                │          │
│         └────────── results/ ◄───────────┘          │
│                                                     │
│   ideas.md ◄── research ── report.md                │
└──────────────────────┬──────────────────────────────┘
                       │ monitors & heals
               ┌───────▼────────┐
               │  bug_fixer.py  │
               │   (watchdog)   │
               └────────────────┘
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

# 4. Run (always start both)
nohup python orze/farm.py -c orze.yaml >> results/farm.log 2>&1 &
nohup python orze/bug_fixer.py -c orze.yaml >> results/bug_fixer.log 2>&1 &
```

## Key Features

- **Agent roles** — multiple agents (research, documenter, analyzer) run alongside training, each with their own cooldown and state
- **LLM-agnostic** — built-in Claude Code support (`mode: claude`), or bring any LLM via `mode: script` (GPT, Gemini, local models, custom pipelines)
- **Multi-GPU** — claims and trains across all GPUs in parallel
- **Parallel eval** — eval scripts launch non-blocking so GPUs stay busy
- **Health monitoring** — stall detection, OOM detection, disk space checks
- **Push notifications** — Telegram, Slack, Discord, or any webhook. Every message includes the top 10 leaderboard
- **Telegram bot** — text back to the bot in natural language to check status, query results, or add ideas from your phone
- **Configurable report** — custom columns, metrics from any JSON file
- **Multi-machine** — works across machines on shared filesystems (NFS/EFS/FSx)
- **Failure handling** — auto-skip after N failures, orphan cleanup
- **Self-healing** — companion `bug_fixer.py` watchdog runs alongside farm.py, auto-restarts crashed processes, kills stuck jobs, and spawns an LLM to diagnose and patch farm.py bugs in real time
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

### Telegram Bot

Text back to the Telegram bot in natural language. The companion `bot.py` process receives your messages and routes them to an LLM that has full context of your orze system.

Built-in commands respond instantly (no LLM): `/status`, `/top`, `/help`, `/ping`. Everything else goes to the LLM agent.

```bash
# Start alongside farm.py
nohup python orze/bot.py -c orze.yaml >> results/bot.log 2>&1 &
```

Uses the same `bot_token`/`chat_id` from your notifications config. Optional dedicated config:

```yaml
telegram_bot:
  mode: claude              # or "script" for GPT/Gemini/local
  model: sonnet
  timeout: 120
  rate_limit: 10            # max messages per minute
```

## Self-Healing

Orze ships with `bug_fixer.py` — a lifetime watchdog that runs alongside `farm.py` and keeps the system healthy:

- **Auto-restart** — if farm.py dies, the watchdog detects it within 60s and restarts it
- **Stuck process killer** — training/eval jobs with no CPU activity past timeout get killed automatically
- **Zombie reaper** — detects and cleans up defunct processes
- **Stale claim cleanup** — flags ideas claimed but never completed by crashed workers
- **Disk space monitoring** — alerts when free space drops below threshold
- **LLM-powered code fixes** — for errors in farm.py itself, spawns a Claude session to diagnose and patch the bug (local commit only, human reviews before push)

The watchdog **only touches orze platform code** (`farm.py`). It never modifies your training scripts, configs, or data.

```bash
# Launch all companions together
nohup python orze/farm.py -c orze.yaml >> results/farm.log 2>&1 &
nohup python orze/bug_fixer.py -c orze.yaml >> results/bug_fixer.log 2>&1 &
nohup python orze/bot.py -c orze.yaml >> results/bot.log 2>&1 &  # optional
```

Optional tuning in `orze.yaml`:

```yaml
bug_fixer:
  check_interval: 60        # seconds between checks
  stale_training_min: 45    # kill idle training after N minutes
  stale_eval_min: 60        # kill idle eval after N minutes
  max_fixes_per_hour: 3     # rate limit on LLM fix sessions
```

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
  --stop                    Gracefully stop a running instance
  --disable                 Stop and persistently disable Orze (survives restarts)
  --enable                  Remove persistent disable flag to allow Orze to run
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

## Stopping Orze

Three levels of stopping, from temporary to persistent:

```bash
# 1. Stop running instances (one-time, cleared on next startup)
python orze/farm.py -c orze.yaml --stop

# 2. Disable persistently (survives restarts, blocks bug_fixer auto-restart)
python orze/farm.py -c orze.yaml --disable

# 3. Re-enable after disable
python orze/farm.py -c orze.yaml --enable
```

**How it works across machines:**

| Command | Mechanism | Scope | Persists? |
|---------|-----------|-------|-----------|
| `--stop` | Writes `.orze_stop_all` + SIGTERM local PIDs | All machines on shared FS | No — cleared on next startup |
| `--disable` | Writes `.orze_disabled` + `.orze_stop_all` | All machines on shared FS | Yes — must `--enable` to remove |
| `Ctrl+C` / `kill -15` | Signal handler | Local machine only | No |

The shared filesystem (NFS/EFS/FSx) is the coordination layer. Running instances poll for stop/disable files every `poll` seconds (default 30s). `bug_fixer.py` also respects `.orze_disabled` and will not auto-restart a disabled Orze.

## License

Apache 2.0
