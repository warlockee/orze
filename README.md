# orze

A decentralized, agentic experiment orchestrator for ML training. 

Orze runs the full research loop: **generate ideas → train → evaluate → learn → repeat**. It coordinates GPUs via filesystem locks (`mkdir`), works across machines, and supports any LLM as the research agent — Claude, GPT, Gemini, local models, or your own script.

**Website:** [orze.ai](https://orze.ai)

## Install

```bash
# One-line install (installs uv if needed, then orze):
curl -sL https://orze.ai/setup.sh | bash

# Or with a specific project path:
curl -sL https://orze.ai/setup.sh | bash -s /path/to/shared/storage/project

# Manual install with uv (recommended):
uv tool install orze

# Or with pip:
pip install orze
```

## Quick Start
**If you are in Claude/Gemini/Codex Cli**
```bash
do @ORZE-AGENT.md
```
**If not**
```bash
orze
```

That's it. Orze will auto-detect your GPUs and start running experiments from `ideas.md`. 

## Multi-node
As long as you start the orze in the same shared folder (e.g. /nfs/project-52h/) on any machine, the node will automatically join the research pool. **Orze can auto-update. If one machine is updated, everyone else will auto update in 15mins**

## Key Features

- **Scales to 1M+ Experiments** — SQLite-backed job queue and indexed reporting with O(log N) scheduling.
- **Multi-LLM Research Army** — Run Claude, Gemini, GPT, and local models as parallel research agents. Auto-discovers API keys from your environment.
- **Config Inheritance** — Child ideas inherit parent configs; research agents specify only what changes.
- **Circuit Breaker** — Stops the node if failure rates spike. Schema validation catches hallucinations before they hit GPUs.
- **Self-Healing Watchdog** — Companion `bug_fixer` agent auto-restarts crashed processes, kills stuck jobs, and diagnoses errors using an LLM.
- **Multi-Machine** — Orchestrate thousands of GPUs across nodes via shared filesystems (NFS/EFS/FSx).
- **HP Sweep** — List-valued hyperparameters (e.g. `lr: [1e-4, 3e-4]`) auto-expand into Cartesian product sub-runs.
- **Admin Panel** — Real-time web dashboard auto-starts at `http://localhost:8787` when orze launches.
- **Clean Uninstall** — `orze --uninstall` removes all runtime files and the package itself, preserving only your research results.

## How It Works

```
┌─────────────────────────────────────────────────────┐
│                       orze                          │
│                                                     │
│   ┌───────────┐     ┌─────────┐     ┌──────────┐    │
│   │ Research  │────>│  Train  │────>│ Evaluate │    │
│   │ (any LLM) │     │ (GPUs)  │     │          │    │
│   └─────▲─────┘     └─────────┘     └──────────┘    │
│         │                                │          │
│         └────────── results/ ◄───────────┘          │
│                                                     │
│   ideas.md ◄── research ── report.md                │
└─────────────────────────────────────────────────────┘
```

## Admin Panel

When orze starts, it automatically launches a real-time admin dashboard on **http://localhost:8787**. No extra install or setup needed.

<img width="900" height="674" alt="admin-panel" src="https://github.com/user-attachments/assets/b23879e3-d064-4e02-8251-6e8dbfad21f9" />
<img width="900" height="674" alt="admin-queue" src="https://github.com/user-attachments/assets/39747da2-7b7f-4a9f-ad4a-7cfaca41407b" />
<img width="900" height="551" alt="admin-leaderboard" src="https://github.com/user-attachments/assets/70e77941-efbf-4018-9200-93ea77998c5e" />

The panel provides:
- **Overview** — GPU utilization, VRAM, temperature, queue depth, top results at a glance
- **Nodes** — Per-host heartbeat status, free GPUs, active runs across your cluster
- **Runs** — All active, completed, and failed experiments with logs and metrics
- **Queue** — Pending ideas waiting to be scheduled
- **Leaderboard** — Ranked results sorted by your primary metric
- **Alerts** — Failure spikes, stale nodes, disk warnings
- **Settings** — Live view of your `orze.yaml` configuration

To change the port, set the `ORZE_ADMIN_PORT` environment variable:
```bash
ORZE_ADMIN_PORT=9000 orze
```

## Telegram Notifications

Get real-time alerts on your phone when experiments complete, fail, or hit a new best score.

**1. Create a bot** — message [@BotFather](https://t.me/BotFather) on Telegram:
```
/newbot
```
Follow the prompts. You'll receive a bot token like `123456789:ABCdefGHIjklMNOpqrsTUVwxyz`.

**2. Get your chat ID** — message [@userinfobot](https://t.me/userinfobot) on Telegram. It will reply with your numeric chat ID.

**3. Add to `orze.yaml`:**
```yaml
notifications:
  enabled: true
  on: [completed, failed, new_best]
  channels:
    - type: telegram
      bot_token: "YOUR_BOT_TOKEN"
      chat_id: "YOUR_CHAT_ID"
```
<img width="521" height="341" alt="tg" src="https://github.com/user-attachments/assets/f931221d-b428-4b85-9a8e-af6d516cb5ad" />

Notification events:
- `completed` — an experiment finished successfully
- `failed` — an experiment errored out
- `new_best` — a new top score on your primary metric

## Service Management (Watchdog)

Built-in process supervision that auto-restarts orze if it crashes or stalls. Works across multi-node clusters — each node manages itself independently.

```bash
orze service install -c orze.yaml    # Install watchdog (crontab or systemd)
orze service status                  # Check watchdog & process health
orze service logs                    # View watchdog log
orze service uninstall               # Remove watchdog
```

**How it works:**
- Checks every minute (crontab) or uses systemd `Restart=on-failure`
- Detects stale processes via heartbeat age (default: 30min threshold)
- Respects `--stop` and `--disable` flags — won't restart if you intentionally stopped orze
- Sends notifications on restart (Telegram/Slack/Discord) if configured
- Per-node config stored in `~/.orze_service.json` — safe for shared filesystems

**Install methods** (auto-detected):
- `systemd` — preferred; uses `systemctl --user` (no root needed)
- `crontab` — fallback; adds a 1-minute check

```bash
# Force a specific method
orze service install -c orze.yaml --method crontab

# Custom stall threshold (seconds)
orze service install -c orze.yaml --stall-threshold 3600
```

Add `watchdog_restart` to your notification events to get alerts:
```yaml
notifications:
  enabled: true
  on: [completed, failed, new_best, watchdog_restart]
```

## The Contract

Your training script receives these standard arguments:
```bash
python train.py --idea-id idea-001 --results-dir results --ideas-md ideas.md --config base.yaml
```

**Required Output:** Write `results/{idea_id}/metrics.json`:
```json
{"status": "COMPLETED", "test_accuracy": 0.92, "training_time": 142.5}
```

See [**RULES.md**](src/orze/RULES.md) for the full technical specification.

## License

Apache 2.0
