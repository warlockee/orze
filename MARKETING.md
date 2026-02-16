# Orze — Marketing & Product Brief

> Source of truth for the orze.ai frontend. Every section maps to a page or component.

---

## Hero

**Headline:** Auto-Research on Autopilot

**Subheadline:** One script, all GPUs, any LLM. Orze runs the full research loop — generate ideas, train, evaluate, learn, repeat — while you sleep.

**CTA:** Get Started (free, open source)

**Install:**
```bash
curl -sL https://raw.githubusercontent.com/warlockee/orze/main/setup.sh | bash
```

---

## The Problem

Running ML experiments is manual, slow, and wasteful:

- **GPU babysitting** — You watch `nvidia-smi`, manually launch the next run, and hope nothing crashes overnight
- **Lost progress** — A hung process eats your GPU for 8 hours. A crash loses your best checkpoint
- **Scattered results** — Spreadsheets, wandb tabs, terminal scrollback. Which run was the best again?
- **Research bottleneck** — You can only think of new experiments as fast as you can read old results
- **Scaling pain** — Adding a second machine means setting up job queues, databases, coordinators

---

## The Solution

Orze is a single Python script that turns your GPU cluster into an autonomous research lab.

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

Define experiments in Markdown. Orze claims GPUs, launches training, monitors health, runs evaluation, updates the leaderboard, and notifies you on Slack. An LLM agent reads results and generates the next batch of ideas. The loop runs 24/7.

---

## How It Works (Step by Step)

1. **Research** — An LLM agent (Claude, GPT, Gemini, or your own script) reads the leaderboard and generates new experiment ideas
2. **Parse** — Orze reads `ideas.md` for experiment definitions with YAML config blocks
3. **Claim** — Each idea is claimed via atomic `mkdir` — no race conditions, even across machines
4. **Train** — Training launches as a subprocess on a free GPU with `CUDA_VISIBLE_DEVICES`
5. **Monitor** — Health checks detect stalls, OOM, timeouts, and disk pressure
6. **Evaluate** — Post-training eval runs non-blocking so GPUs stay busy
7. **Notify** — Push updates to Telegram, Slack, Discord with the top 10 leaderboard
8. **Report** — Auto-generated markdown leaderboard + machine-readable `status.json`
9. **Repeat** — Sleep, then start again

---

## Key Features

### GPU Orchestration
- Auto-detects all GPUs, claims via filesystem locks
- Parallel training across all available GPUs
- No resource manager, no job queue — just files
- Works on 1 GPU or 100

### LLM-Powered Research
- Built-in Claude Code integration (zero-config)
- Or use any LLM via script mode — GPT, Gemini, Llama, Mistral
- Agent reads results and autonomously generates next experiments
- Multiple agent roles: research, documenter, analyzer, or custom

### Health Monitoring
- Stall detection — kills hung processes automatically
- OOM detection — catches CUDA out of memory
- Timeout enforcement — no experiment runs forever
- Disk space protection — pauses launches when storage is low
- Orphan cleanup — reclaims abandoned GPU claims

### Push Notifications
- Telegram, Slack, Discord, or any webhook
- Events: experiment completed, failed, new best, periodic report
- Every notification includes the top 10 leaderboard
- Per-channel event filtering

### Multi-Machine
- Works across any shared filesystem (NFS, EFS, FSx)
- Per-host heartbeats, merged status view
- No central coordinator — each machine is independent
- Add a machine by starting `farm.py`. That's it.

### Crash Recovery
- State persisted every iteration
- Resume from exactly where you left off
- Failed experiments auto-retry with configurable limits
- Graceful shutdown on SIGINT/SIGTERM

---

## The Contract

Your training script gets:
```
CUDA_VISIBLE_DEVICES=N python train.py \
  --idea-id idea-001 \
  --results-dir results \
  --ideas-md ideas.md \
  --config configs/base.yaml
```

Your script writes:
```json
{"status": "COMPLETED", "test_accuracy": 0.92, "training_time": 142.5}
```

That's the entire integration. Any framework. Any language (wrap it). Any metric.

---

## Quick Start

### With Claude Code (recommended)

```bash
# Install orze
curl -sL https://raw.githubusercontent.com/warlockee/orze/main/setup.sh | bash

# Open Claude Code and say:
@orze/AGENT.md set up and run experiments for this project
```

Claude explores your codebase, creates the config, writes seed ideas, and launches the loop. Running in ~10 minutes.

### Manual Setup

```bash
# Install
curl -sL https://raw.githubusercontent.com/warlockee/orze/main/setup.sh | bash

# Configure
cp orze/orze.yaml.example orze.yaml
# Edit orze.yaml — set train_script, python path, report columns

# Write seed experiments
# Edit ideas.md — see RULES.md for the format

# Run
python orze/farm.py -c orze.yaml
```

---

## Configuration

Everything lives in one YAML file:

```yaml
# Core
train_script: train.py
ideas_file: ideas.md
base_config: configs/base.yaml
results_dir: results
python: /path/to/venv/bin/python3

# Timeouts & health
timeout: 3600
stall_minutes: 30
max_idea_failures: 3
min_disk_gb: 20

# Research agent (any LLM)
roles:
  research:
    mode: claude          # or mode: script for GPT/Gemini/local
    rules_file: RESEARCH_RULES.md
    model: sonnet
    cooldown: 300

# Notifications
notifications:
  enabled: true
  channels:
    - type: telegram
      bot_token: "your-token"
      chat_id: "your-chat-id"

# Leaderboard
report:
  primary_metric: test_accuracy
  sort: descending
  columns:
    - {key: "test_accuracy", label: "Accuracy", fmt: ".4f"}
    - {key: "training_time", label: "Time(s)", fmt: ".0f"}
```

Full reference: [orze.yaml.example](https://github.com/warlockee/orze/blob/main/orze.yaml.example)

---

## Agent Modes

### mode: claude
Zero-config for Claude Code users. Orze calls the `claude` CLI with your rules file as the prompt. Supports model selection, tool restrictions, and custom args.

```yaml
roles:
  research:
    mode: claude
    rules_file: RESEARCH_RULES.md
    model: sonnet
    cooldown: 300
```

### mode: script
Bring any LLM. Write a Python script that calls your preferred API, reads results, and appends ideas to `ideas.md`. Orze handles scheduling, locking, and logging.

```yaml
roles:
  research:
    mode: script
    script: my_gpt_researcher.py
    args: ["--results-dir", "{results_dir}", "--ideas", "{ideas_file}"]
    cooldown: 300
    env:
      OPENAI_API_KEY: sk-...
```

### Multiple Roles
Run different agents for different tasks:

```yaml
roles:
  research:
    mode: claude
    rules_file: RESEARCH_RULES.md
    cooldown: 300
  analyzer:
    mode: script
    script: pattern_detector.py
    cooldown: 600
  documenter:
    mode: claude
    rules_file: DOCUMENTER_RULES.md
    model: haiku
    cooldown: 900
```

---

## Experiments as Markdown

Ideas live in a plain markdown file. Human-readable, Git-friendly, append-only.

```markdown
## idea-001: Simple CNN Baseline
- **Priority**: medium
- **Category**: architecture
- **Hypothesis**: Establish baseline with a simple 3-layer CNN

\```yaml
model:
  type: simple_cnn
  channels: [32, 64, 128]
training:
  epochs: 20
  lr: 0.001
\```

## idea-002: ResNet with Skip Connections
- **Priority**: high
- **Category**: architecture
- **Hypothesis**: Skip connections should improve gradient flow

\```yaml
model:
  type: resnet_small
  channels: [64, 128]
  blocks_per_stage: 3
training:
  epochs: 30
  lr: 0.0005
\```
```

The LLM agent appends new ideas here. You can also add ideas manually. Orze picks them up on the next cycle.

---

## Notifications

### Telegram
```yaml
notifications:
  enabled: true
  channels:
    - type: telegram
      bot_token: "123456:ABC-DEF"
      chat_id: "-100123456789"
```

### Slack
```yaml
    - type: slack
      webhook_url: "https://hooks.slack.com/services/T.../B.../xxx"
      on: [new_best, failed]    # only notify on these events
```

### Discord
```yaml
    - type: discord
      webhook_url: "https://discord.com/api/webhooks/123/abc"
```

### Generic Webhook
```yaml
    - type: webhook
      url: "https://your-api.com/hook"
      headers:
        Authorization: "Bearer token"
```

Events: `completed`, `failed`, `new_best`, `report`

---

## Comparison

| | Orze | Ray Tune | W&B Sweeps | SageMaker |
|---|---|---|---|---|
| Setup time | 5 min | 30 min | 15 min | 1+ hr |
| External services | None | Ray cluster | W&B cloud | AWS |
| LLM agent | Built-in | No | No | No |
| Multi-machine | Shared filesystem | Ray cluster | Cloud | AWS |
| Config format | YAML + Markdown | Python | YAML | JSON |
| Cost | Free | Free | $50+/mo | Pay per use |
| Notifications | Telegram/Slack/Discord | Limited | Email/Slack | SNS |
| Crash recovery | Automatic | Manual | N/A | Managed |

---

## Who It's For

### ML Researchers
You have 4-8 GPUs and a pile of ideas. You want to run them all overnight and see results in the morning. Orze keeps every GPU busy, kills hung jobs, and shows you a leaderboard when you wake up.

### Startup ML Teams
You can't afford dedicated MLOps. Orze is zero infrastructure — no Redis, no Kubernetes, no cloud dashboard subscriptions. One YAML file, one script, all your GPUs.

### Research Labs
You have a cluster with a shared filesystem. Multiple researchers need to run experiments without stepping on each other. Orze coordinates via filesystem locks — no central scheduler to maintain.

### Solo Developers
You have one machine with one GPU. Orze still helps — it manages the queue, handles crashes, sends you notifications, and (with an LLM agent) generates experiments you wouldn't have thought of.

---

## Pricing

**Free and open source.** Apache 2.0 license.

No cloud account. No API keys (unless you want the LLM agent). No vendor lock-in. Fork it, modify it, ship it.

---

## Links

- **GitHub:** [github.com/warlockee/orze](https://github.com/warlockee/orze)
- **Website:** [orze.ai](https://orze.ai)
- **Install:** `curl -sL https://raw.githubusercontent.com/warlockee/orze/main/setup.sh | bash`
- **License:** Apache 2.0

---

## SEO Keywords

gpu experiment orchestrator, ml experiment automation, auto research, llm research agent, multi-gpu training, experiment tracking, hyperparameter search, claude code ml, autonomous ml research, gpu coordination, distributed training, experiment leaderboard, ml notifications, research automation
