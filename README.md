# orze

[![PyPI](https://img.shields.io/pypi/v/orze)](https://pypi.org/project/orze/)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](LICENSE)
[![PyPI - orze-pro](https://img.shields.io/pypi/v/orze-pro?label=orze-pro)](https://pypi.org/project/orze-pro/)

A GPU experiment orchestrator for ML research.

Orze runs experiments on GPUs: **schedule ideas → train → evaluate → report → repeat**. It coordinates GPUs via filesystem locks, works across machines, and gives you a complete leaderboard, notifications, and analysis — out of the box.

**Website:** [orze.ai](https://orze.ai)

## Install

```bash
# One-line install (installs uv if needed, then orze + project scaffold):
curl -sL https://orze.ai/setup.sh | bash

# Or with pip:
pip install orze
```

### Upgrade to Pro

When you're ready for autonomous research agents:

```bash
pip install orze-pro --extra-index-url https://__token__:${ORZE_PRO_KEY}@pypi.orze.ai/simple/
orze pro activate ORZE-PRO-xxx...
# ✓ Licensed to Acme Corp (pro), expires 2027-12-31

orze pro status                    # verify it worked
```

No config changes — pro features activate automatically. [What's in pro?](#orze-vs-orze-pro)

**License management:**
```bash
orze pro status                    # check license info
orze pro deactivate                # remove license key
```

**Alternative activation methods** (for shared clusters or CI):
```bash
# In your project .env file:
ORZE_PRO_KEY=ORZE-PRO-xxx...

# Or as environment variable:
export ORZE_PRO_KEY=ORZE-PRO-xxx...
```

## orze vs orze-pro

orze is a **complete, production-ready tool**. orze-pro adds **autopilot** — so experiments run while you sleep.

| Feature | orze (free) | + orze-pro |
|---------|:-----------:|:----------:|
| GPU scheduling & multi-node | ✓ | ✓ |
| Idea queue (ideas.md + SQLite) | ✓ | ✓ |
| Hyperparameter sweep (auto-expand grid) | ✓ | ✓ |
| Leaderboard report | ✓ | ✓ |
| Telegram/Slack notifications (rich) | ✓ | ✓ |
| Admin dashboard & MCP server | ✓ | ✓ |
| Retrospection (plateau detection) | ✓ | ✓ |
| Cross-experiment regression analysis | ✓ | ✓ |
| Failure analysis & categorization | ✓ | ✓ |
| Checkpoint GC | ✓ | ✓ |
| Sealed eval protection | ✓ | ✓ |
| Service watchdog (auto-restart) | ✓ | ✓ |
| **Autonomous research agents** (Gemini/GPT/Claude) | | ✓ |
| **Auto-fix failed experiments** | | ✓ |
| **Code evolution on plateau** | | ✓ |
| **Meta-research (strategy adjustment)** | | ✓ |
| **Interactive Telegram/Slack bot** | | ✓ |

### Research Loop Comparison

| | orze free | + orze-pro |
|---|---|---|
| **How ideas are generated** | **Smart Suggestions** — rule-based: detects regressions, generates scale sweeps, perturbations | **Research Agents** — LLM-driven: reads all results, forms hypotheses, designs novel experiments |
| **How failures are handled** | You read the failure log | Auto-fix: LLM diagnoses and patches the error |
| **How plateaus are handled** | Smart Suggestions tries parameter variations | Code Evolution: LLM modifies your train script |
| **Does research stop?** | **Never** — Smart Suggestions keeps GPUs busy | **Never** — agents run indefinitely |
| **Requires API key?** | No | Yes (Gemini/OpenAI/Anthropic) |

### Compatibility

| orze | orze-pro | Notes |
|------|----------|-------|
| 3.0.x | 0.1.x | Current release |

## User Journeys

### Free user — "Smart Suggestions keep my GPUs busy"

```bash
orze init                          # creates orze.yaml, ideas.md, train.py
vim ideas.md                       # add your first experiment ideas
orze -c orze.yaml                  # orze runs them on GPUs

# After your ideas complete, orze doesn't stop:
# → analyzes results, detects regressions and tradeoffs
# → Smart Suggestions auto-generates new ideas:
#   "Fix SPG regression: scale 1.0->0.9"
#   "Tradeoff sweep: scale=0.95"
#   "Push further: scale 1.05 (no regressions)"
# → runs them, analyzes again, generates more
# → you check in when you want, add your own ideas anytime
```

You seed the initial ideas. Orze keeps the loop going with Smart Suggestions. You steer when you want.

### Pro user — "I sleep, orze researches"

```bash
pip install orze-pro --extra-index-url https://__token__:${ORZE_PRO_KEY}@pypi.orze.ai/simple/
orze -c orze.yaml                  # same command — pro features activate automatically
# → research agent reads results and proposes new ideas
# → failed experiments get auto-fixed and retried
# → when stuck on a plateau, code evolution kicks in
# → you wake up to a better model
```

Same `orze.yaml`. Same workflow. Pro just adds autonomy.

### The upgrade moment

Smart Suggestions is finding ±5% scale variations. It's keeping GPUs busy. But you see this in the logs:

```
[Orze Smart Suggestions] Generated 3 ideas from experiment analysis
  → Fix SPG regression: scale 1.0->0.9
  → Tradeoff sweep: scale=0.95
  → Push further: scale 1.05 (no regressions)
```

It's systematic — but it's not *thinking*. It can't reason about *why* SPG regressed (domain mismatch in training data), or propose *adding SPG training data* as a fix, or decide *the scale sweep is exhausted and a new LoRA training run is needed*.

That's when you `pip install orze-pro`. The research agent reads the same insights and generates:

```
## idea-r8a2f1: Train LoRA v8 with SPG data to fix domain mismatch
- **Hypothesis**: SPG regression (+0.66%) is caused by training data lacking
  financial transcript style. Adding 3K SPG samples should reduce insertions.
```

Smart Suggestions explores. Research Agents discover.

## Quick Start

**If you are in Claude/Gemini/Codex CLI:**
```bash
do @ORZE-AGENT.md
```

**If not:**
```bash
orze
```

That's it. Orze auto-detects GPUs and starts running experiments from `ideas.md`.

## File Layout

Orze uses a hidden `.orze/` directory for runtime state and an `orze_results/` directory for research artifacts:

```
your-project/
├── .orze/                    # Runtime state (gitignored)
│   ├── ideas.md              # Research manifest
│   ├── idea_lake.db          # Experiment history
│   ├── logs/                 # Role logs (professor/, engineer/, etc.)
│   ├── receipts/             # Execution evidence per cycle
│   ├── state/                # Daemon PID, version.json, caches
│   ├── rules/                # Custom agent rules (ENGINEER_RULES.md, etc.)
│   ├── triggers/             # One-shot role triggers
│   ├── heartbeats/           # Per-host liveness signals
│   ├── backups/              # Timestamped ideas.md backups
│   ├── feedback/             # Failure feedback summaries
│   └── tmp/                  # Scratch space
├── orze_results/             # Research outputs
│   ├── methods/              # Generated code (per-role, per-cycle)
│   ├── knowledge/            # Analysis insights, retrospection
│   └── stray/                # Quarantined root-polluting files
├── orze.yaml                 # Project config
└── GOAL.md                   # High-level research objective
```

**Migration:**  
Existing projects (pre-v4.1) should run `orze admin migrate` once to upgrade to the new layout. Migration is automatic on first `orze run` if not already applied.

**One-liner upgrade:**
```bash
orze upgrade                  # Reinstall orze + orze-pro and restart daemon
```

## Multi-node

Start orze in the same shared folder (e.g. `/nfs/project-52h/`) on any machine — the node automatically joins the research pool. **Orze can auto-update across nodes.**

## Key Features

- **Scales to 1M+ Experiments** — SQLite-backed job queue with O(log N) scheduling
- **Config Inheritance** — Child ideas inherit parent configs; specify only what changes
- **HP Sweep** — `lr: [1e-4, 3e-4]` auto-expands into all combinations
- **Failure Protection** — Stops automatically when failure rates spike
- **Cross-Experiment Analysis** — Detects regressions, tradeoffs, and suggests actions
- **Rich Notifications** — GPU VRAM, per-dataset breakdown, verified results, target/gap tracking
- **Admin Panel** — Real-time web dashboard at `http://localhost:8787`
- **Clean Uninstall** — `orze --uninstall` removes runtime files, preserves results

## How It Works

Orze runs a continuous loop: pick an idea from the queue, train it on a free GPU, evaluate, record metrics. When ideas run out, orze generates variations of your best configs automatically — **the research never stops**, even without pro.

With **orze-pro**, LLM agents replace parameter variations with intelligent, hypothesis-driven ideas.

## Admin Panel

Auto-launches at **http://localhost:8787**. No extra install needed.

<img width="900" height="674" alt="admin-panel" src="https://github.com/user-attachments/assets/b23879e3-d064-4e02-8251-6e8dbfad21f9" />
<img width="900" height="674" alt="admin-queue" src="https://github.com/user-attachments/assets/39747da2-7b7f-4a9f-ad4a-7cfaca41407b" />
<img width="900" height="551" alt="admin-leaderboard" src="https://github.com/user-attachments/assets/70e77941-efbf-4018-9200-93ea77998c5e" />

## Telegram Notifications

Rich notifications with GPU VRAM, per-dataset breakdown, and target tracking:

```
📊 Orze Status — a100-41
✅ 20 completed | ❌ 0 failed | ⏳ 6 queued | 🔄 4 running
🎯 Verified: 5.43% avg WER | Target: 5.40% | Gap: +0.03%
  AMI=9.8 | E22=9.0 | GS=8.5 | LS-C=1.3 | SPG=3.6 | TED=2.7 | VP=6.1
🖥 GPU0:idle GPU1:18G/80G(51%) GPU3:17G/80G(48%)
🤖 Model: higgs-audio-v3-8b
⏱ Up 2h15m
```

Setup:
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

## Service Management (Watchdog)

```bash
orze service install -c orze.yaml    # auto-restart on crash
orze service status                  # check health
orze service uninstall               # remove
```

## The Contract

Your training script receives:
```bash
python train.py --idea-id idea-001 --results-dir results --ideas-md ideas.md --config base.yaml
```

**Required output:** `results/{idea_id}/metrics.json`:
```json
{"status": "COMPLETED", "test_accuracy": 0.92, "training_time": 142.5}
```

Orze writes `idea_config.yaml` to the results directory before launching, containing the merged base + idea config.

See [**SKILL.md**](SKILL.md) for the full technical specification.

## Citation

```bibtex
@article{li2026autoresearching,
  title={Auto Researching, not hyperparameter tuning: Convergence Analysis of 10,000 Experiments},
  author={Li, Xiaoyi},
  journal={arXiv preprint arXiv:2603.15916},
  year={2026}
}
```

## License

Apache 2.0 — orze is and will always be free and open source.

[orze-pro](https://github.com/warlockee/orze-pro) (autopilot features) is commercially licensed.
