# Twitter/X Thread

## Rules of engagement
- Hook in the first tweet. You have 2 seconds.
- Short sentences. One idea per tweet.
- Visual proof > claims. Include screenshots/GIFs.
- End with a clear CTA.
- No "excited to announce" or "I'm thrilled." Just say what it does.

---

## Thread

### Tweet 1 (Hook)
```
I got tired of manually queuing GPU experiments, so I built a tool that does the entire research loop autonomously.

It generates ideas with LLMs, trains them, evaluates results, learns what works, and repeats.

Open source. Here's how it works:

🧵
```

### Tweet 2 (Problem)
```
The ML research workflow is mostly waiting:

- Launch experiment
- Wait 30 min to 3 hours
- Check results
- Think about what to try next
- Repeat

GPUs sit idle while you eat lunch, sleep, or just don't notice training finished.
```

### Tweet 3 (Solution — keep it concrete)
```
Orze fixes this.

pip install orze
orze --init
orze

It auto-detects your GPUs, picks experiments from a queue, runs training, collects metrics, and moves to the next one.

Your GPUs never sit idle.
```

### Tweet 4 (The killer feature)
```
But the real power is the research loop.

Add an LLM research agent (Gemini, GPT-4o, Claude, or local Ollama) and Orze closes the loop:

Agent generates ideas → GPUs train them → Results update leaderboard → Agent reads results → Generates better ideas → Repeat

You can walk away.
```

### Tweet 5 (Proof — Kaggle story)
```
I used it for a Kaggle competition.

Pointed it at 8 H100s on a Friday. By Monday it had:

- Run 800+ experiments autonomously
- Discovered a training strategy I wouldn't have tried
- Achieved competitive results

The breakthrough idea came from the LLM agent analyzing patterns across hundreds of experiments.
```

### Tweet 6 (Architecture — for the technical crowd)
```
Architecture is deliberately simple:

- Ideas: Markdown file (human-readable)
- Results: JSON files (one per experiment)
- Coordination: filesystem locks (no Redis, no K8s)
- Multi-node: mount shared filesystem, run orze on each machine

Works on any cluster with GPUs and NFS.
```

### Tweet 7 (Features — quick hits)
```
Other things it does:

- Hyperparameter sweeps (grid, random, TPE)
- Telegram/Slack/Discord notifications
- Admin dashboard
- Watchdog auto-restart
- Goal-driven research (write a GOAL.md, agent uses it as context)
- Automatic garbage collection
- Multi-node clustering
```

### Tweet 8 (CTA)
```
Apache 2.0 licensed. pip install orze.

GitHub: github.com/erikhenriksson/orze
Docs: orze.ai

If you have GPUs sitting idle between experiments, give it a try.
```

---

## Standalone tweets (for drip posting over weeks)

### Standalone 1 — GPU utilization angle
```
Most research labs use their GPUs less than 40% of the time.

Not because they don't have work — because the loop between "results came in" and "next experiment starts" has a human in it.

We built Orze to remove that bottleneck.

pip install orze
github.com/erikhenriksson/orze
```

### Standalone 2 — LLM agent angle
```
What if your LLM research agent could:

1. Read your completed experiment results
2. Analyze what worked and what didn't
3. Generate new experiment ideas
4. Queue them for training
5. Repeat

That's what Orze does. Open source.

github.com/erikhenriksson/orze
```

### Standalone 3 — Simplicity angle
```
Our GPU experiment orchestrator has:

- No Kubernetes
- No Docker
- No database server
- No message queue
- No cloud vendor lock-in

Just a Python package, a YAML file, and filesystem locks.

Sometimes the right architecture is the simple one.

pip install orze
```

### Standalone 4 — After getting results/traction
```
[X] days of running Orze autonomously:

- [N] experiments completed
- [N] GPUs at [X]% utilization
- Best model found at experiment [N] (would have taken weeks manually)

The LLM research agent generated [N]% of the ideas that beat the baseline.

Open source: github.com/erikhenriksson/orze
```
