# We Built an Autopilot for GPU Experiments

**TL;DR:** Orze is an open-source GPU experiment orchestrator that runs the full ML research loop autonomously — generating hypotheses with LLM agents, training models, evaluating results, learning what works, and repeating. One YAML file, no Kubernetes, works across multiple machines with a shared filesystem.

---

## The Problem

If you run ML experiments on GPUs, your workflow probably looks like this:

1. Come up with an idea
2. Write a config
3. Launch training
4. Wait 30 minutes to 3 hours
5. Check results
6. Think about what to try next
7. Repeat

Most of the time, your GPUs are idle — waiting for you to wake up, finish lunch, or just notice that training finished.

We had the same problem. Eight H100s, and they spent most of their time doing nothing while we manually iterated on ideas.

So we built Orze.

## What Orze Does

Orze is a lightweight, filesystem-based experiment orchestrator. You give it:

- A training script
- Access to GPUs
- An `orze.yaml` config file

And it handles the rest:

```bash
pip install orze
orze --init    # generates config, ideas.md, train.py scaffold
orze           # starts orchestrating
```

Orze auto-detects your GPUs, picks unclaimed experiments from your idea queue, runs training, collects metrics, updates a leaderboard, and moves on. Your GPUs never sit idle.

But the real power is the **research loop**.

## LLM Research Agents

Orze ships with built-in LLM research agents that close the loop between "results came in" and "what do we try next."

You configure a research role in your `orze.yaml`:

```yaml
roles:
  research_gemini:
    mode: research
    backend: gemini
    model: gemini-2.5-pro
    cooldown: 600
```

The agent reads your completed experiments, analyzes what worked and what didn't, generates new hypothesis-driven ideas, and appends them to your experiment queue. While your GPUs train the current batch, the agent is already planning the next one.

It supports Gemini, OpenAI, Anthropic, local Ollama models, or any OpenAI-compatible endpoint. You can also use Claude Code as a research agent with `mode: claude` for maximum autonomy — it can read your codebase, modify training scripts, and generate ideas with full context.

## The Closed Loop

This is what makes Orze different from experiment trackers like W&B or MLflow. Those are fantastic tools for logging and visualizing — but they don't *run* anything. You still need to manually decide what to try next.

Orze's loop looks like this:

```
LLM Agent generates ideas
    → Orze queues them
        → GPUs train them
            → Evaluator scores them
                → Results update leaderboard
                    → LLM Agent reads results
                        → Generates better ideas
                            → Repeat
```

You can walk away. Come back in the morning. Check your Telegram notifications. The system keeps running.

## How We Used It: The Kaggle Story

We competed in the Nexar Dashcam Collision Prediction challenge on Kaggle. Instead of manually iterating on models, we pointed Orze at our 8 H100s with a research agent and let it run.

Over a weekend, Orze autonomously:
- Generated and trained hundreds of experiments
- Discovered that time-to-event matched training dramatically improved predictions
- Found optimal ensemble combinations through systematic exploration

The time-to-event insight came from the LLM research agent analyzing patterns across completed experiments. It noticed that models consistently struggled with certain temporal patterns and proposed a targeted fix — not something we would have prioritized manually.

## Architecture: Deliberately Simple

Orze is filesystem-based. No database server, no message queue, no Kubernetes, no Docker required.

- **Ideas**: `ideas.md` (Markdown, append-only, human-readable)
- **Results**: `results/{idea-id}/metrics.json` (one directory per experiment)
- **History**: `idea_lake.db` (SQLite, auto-managed)
- **Config**: `orze.yaml` (single file, everything in one place)

This means it works on any machine with GPUs and a shared filesystem. Multi-node? Just mount the same filesystem and run `orze` on each node — they coordinate automatically via filesystem locks.

```
Node 1: orze -c orze.yaml --gpus 0,1,2,3
Node 2: orze -c orze.yaml --gpus 0,1,2,3
```

Eight GPUs become sixteen. No setup, no configuration changes.

## Features at a Glance

- **Auto GPU detection**: No manual assignment. Orze finds free GPUs and claims them.
- **Hyperparameter sweeps**: Grid sweeps expanded inline from ideas.md.
- **Multi-backend research agents**: Gemini, OpenAI, Anthropic, Ollama, custom endpoints.
- **Notifications**: Telegram, Slack, Discord, and generic webhooks.
- **Admin dashboard**: Real-time web UI at `orze --admin`.
- **Watchdog service**: Auto-restarts on crashes. Install with `orze service install`.
- **Goal-driven onboarding**: Write a `GOAL.md` describing what you want to achieve. The research agent uses it as context.
- **Retrospection**: Periodic analysis of what's working and what isn't, fed back into the research loop.
- **Garbage collection**: Automatic cleanup of old checkpoints to manage disk space.
- **Config validation**: `orze --check` catches problems before you waste GPU hours.

## Limitations

To be upfront about what Orze isn't:

- **Single-GPU experiments**: Orze orchestrates many single-GPU experiments across your GPUs. It's not a distributed training framework — if you need data-parallel training across GPUs, that's your training script's job.
- **Filesystem scalability**: Coordination via filesystem locks works well for 1-4 nodes. Beyond that, you may hit filesystem performance limits depending on your storage backend.
- **LLM agent quality**: The research agent generates plenty of mediocre ideas, especially in early cycles. The value comes from volume — the cost of a bad idea is one short training run, but good ideas compound.

## Getting Started

```bash
pip install orze
mkdir my-project && cd my-project
orze --init
# Edit orze.yaml to point at your training script
# Add your first ideas to ideas.md
orze
```

That's it. Your GPUs are now orchestrated.

## What's Next

Orze is Apache 2.0 licensed and fully open source. We're actively developing:

- Team collaboration features
- Advanced analytics and GPU utilization reports
- Cloud provider templates (AWS, GCP, Lambda Labs)
- More research agent capabilities

We'd love your feedback. Try it out, file issues, or just star the repo if this sounds useful.

**GitHub**: https://github.com/warlockee/orze
**PyPI**: https://pypi.org/project/orze/
**Website**: https://orze.ai

---

*If you have GPUs sitting idle between experiments, give Orze a try. Your GPUs will thank you.*
