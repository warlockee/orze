# Hacker News — Show HN Post

## Rules of engagement
- HN hates marketing speak. No superlatives, no emojis, no "revolutionary."
- Lead with the technical problem. Be understated.
- Comments matter more than the post. Be ready to answer deep technical questions.
- Link directly to GitHub, not a landing page.

---

## Title

```
Show HN: Orze – Open-source GPU experiment orchestrator with LLM research agents
```

## Body

```
I built Orze because I was tired of manually babysitting GPU experiments.

The typical ML research loop is: come up with idea → write config → launch training → wait → check results → think about what to try next → repeat. Most of the time your GPUs sit idle waiting for you.

Orze automates the full loop. You give it a training script, a YAML config, and access to GPUs. It auto-detects free GPUs, picks experiments from a queue, runs training, collects metrics, and updates a leaderboard. When you add an LLM research agent (Gemini, OpenAI, Anthropic, or local Ollama), it closes the loop — the agent reads completed results, generates new experiment ideas, and queues them automatically.

No Kubernetes, no Docker, no database server. It's filesystem-based — ideas live in a Markdown file, results in JSON files, coordination via filesystem locks. Multi-node is just "mount the same filesystem and run orze on each machine."

I tested it during a Kaggle competition (Nexar collision prediction) — left it running on 8 H100s over a weekend. It trained hundreds of experiments without intervention. The most interesting outcome: the LLM agent identified a temporal weighting strategy by correlating failure modes across experiments, something I hadn't considered.

Technical details:
- Pure Python, pip install
- Filesystem-based coordination (no message queue, no Redis)
- Supports grid hyperparameter sweeps
- Built-in admin UI (FastAPI)
- Watchdog service for auto-restart
- Notifications to Telegram/Slack/Discord
- Apache 2.0 license

GitHub: https://github.com/warlockee/orze
PyPI: pip install orze

Happy to answer questions about the architecture or the research loop.
```

---

## Anticipated HN questions and answers

**Q: "How is this different from just a bash loop?"**
A: The orchestration (GPU detection, failure handling, watchdog, leaderboard) saves a lot of plumbing. But the real difference is the LLM research agent that closes the loop — it reads results and generates new ideas. A bash loop doesn't learn from its experiments.

**Q: "How does it handle distributed training?"**
A: Orze orchestrates *single-GPU experiments* across multiple GPUs and nodes. Each experiment gets one GPU. It's not doing distributed data-parallel — it's maximizing utilization by keeping all GPUs busy with different experiments. For DDP you'd still use your training script's built-in distributed support.

**Q: "Why filesystem-based instead of a proper database?"**
A: Simplicity and portability. SQLite is used for the idea lake (historical data), but the active coordination is pure filesystem — works on NFS, EFS, FSx, any shared mount. No daemon to keep running, no port conflicts, no schema migrations. It's ugly but it works reliably across heterogeneous clusters.

**Q: "What prevents the LLM agent from generating garbage ideas?"**
A: It reads the full leaderboard, completed experiment configs, and a retrospection analysis. Bad ideas get trained and score poorly — the agent sees that and avoids similar approaches. It's not perfect, but the cost of a bad idea is just one wasted training run (typically 10-30 minutes). Over hundreds of experiments, the signal-to-noise ratio is surprisingly good.

**Q: "Why not use Ray Tune / Optuna?"**
A: Those optimize hyperparameters for a fixed model architecture. Orze explores the *idea space* — different architectures, loss functions, augmentation strategies, feature combinations. The LLM agent generates structurally different experiments, not just parameter variations.
