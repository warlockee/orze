# Product Hunt Launch

## Rules of engagement
- PH audience is broader — founders, PMs, developers, not just ML researchers
- Lead with the outcome, not the tech
- Keep it visual — GIFs and screenshots matter a lot
- The tagline is everything. Max 60 chars.

---

## Listing

### Name
```
Orze
```

### Tagline (max 60 chars)
```
The autopilot for GPU experiments
```

### Description
```
Orze is an open-source GPU experiment orchestrator that runs the full ML research loop autonomously.

Give it GPUs and a research question. It generates experiment ideas using LLM agents, trains them, evaluates results, learns what works, and generates better ideas. Your GPUs never sit idle.

How it works:
1. pip install orze && orze --init
2. Configure your training script and research goal
3. Run orze
4. Walk away — check results on Telegram/Slack/Discord

Key features:
• Auto GPU detection — finds and claims free GPUs automatically
• LLM research agents — Gemini, GPT-4, Claude, or local Ollama
• Multi-node clusters — just mount a shared filesystem
• Admin dashboard — real-time web UI
• Hyperparameter sweeps — grid, random, Bayesian
• Notifications — Telegram, Slack, Discord, and more
• Zero infrastructure — no Docker, no Kubernetes, pure Python

We used it to run 800+ experiments autonomously for a Kaggle competition, discovering training strategies that we wouldn't have found manually.

Open source (Apache 2.0). Free forever.
```

### First Comment (as maker)
```
Hey Product Hunt!

I built Orze because I was frustrated watching my GPUs sit idle between experiments. The bottleneck in ML research isn't training time — it's the gap between "results came in" and "I decided what to try next."

Orze fills that gap with LLM agents that continuously generate new experiment ideas from your results. It's been running our 8-GPU cluster at >90% utilization for months.

The architecture is deliberately simple: Markdown files for ideas, JSON for results, filesystem locks for coordination. No infrastructure to maintain.

Would love your feedback — especially if you work with GPUs and have felt this pain!
```

### Topics
```
Artificial Intelligence, Developer Tools, Open Source, Machine Learning
```

### Media checklist
- [ ] Hero image: terminal showing orze status output + Telegram notification
- [ ] GIF 1: `orze --init` → first experiment launching (30 sec)
- [ ] GIF 2: Admin dashboard showing live leaderboard
- [ ] Screenshot: Telegram notification showing heartbeat status
- [ ] Screenshot: ideas.md with LLM-generated experiments
