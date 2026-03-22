# Reddit r/LocalLLaMA Post

## Rules of engagement
- This community is enthusiast-driven. They love self-hosted, local-first tools.
- Lead with the LLM agent angle — that's what this sub cares about.
- Mention Ollama support prominently.
- Be casual. This isn't an academic sub.

---

## Title

```
Built an open-source tool that uses local LLMs (Ollama) to autonomously generate and run ML experiments on your GPUs
```

## Body

```
I built Orze — a GPU experiment orchestrator that can use a local LLM (via Ollama) to continuously generate new experiment ideas, train them, evaluate results, and feed the results back to the LLM for the next round.

Basically: point it at your GPUs, give it a research question, and walk away. It runs the full loop.

**How the LLM part works**

In your config you set up a research role:

```yaml
roles:
  research_local:
    mode: research
    backend: ollama
    model: llama3
    endpoint: http://localhost:11434
    cooldown: 600
```

Every 10 minutes (configurable), the agent:
1. Reads your completed experiment results and leaderboard
2. Analyzes what worked and what didn't
3. Generates new experiment ideas with specific configs
4. Appends them to a Markdown queue
5. Orze picks them up and trains them on free GPUs

You can also use Gemini, OpenAI, Anthropic, or any OpenAI-compatible API — but the Ollama support means everything stays on your machine. No API keys, no rate limits, no data leaving your network.

**What Orze handles beyond the LLM stuff**

- Auto-detects free GPUs, keeps them all busy
- HP sweeps (grid, random, TPE/Bayesian)
- Multi-node clustering (shared filesystem, no setup)
- Notifications (Telegram, Slack, Discord)
- Admin dashboard (web UI)
- Watchdog service (auto-restart on crashes)
- Automatic checkpoint cleanup

**The architecture is dead simple**

No Docker, no Kubernetes, no database server. Ideas are a Markdown file. Results are JSON. Coordination is filesystem locks. Multi-node is "mount the same NFS share and run orze on each machine."

```bash
pip install orze
orze --init
# edit orze.yaml
orze
```

**Does the local LLM actually generate good ideas?**

Honestly, it depends on the model and the problem. With larger models (70B+, or cloud APIs like Gemini 2.5 Pro) the ideas are surprisingly good — in a Kaggle competition, the agent found a training strategy across 800 experiments that I wouldn't have tried manually.

With smaller local models (7-13B), the ideas are more hit-or-miss, but the cost of a bad idea is just one wasted training run. Over hundreds of experiments, even a noisy idea generator produces useful discoveries through sheer volume.

**Links**

- GitHub: https://github.com/erikhenriksson/orze
- `pip install orze`
- Apache 2.0

Would love feedback, especially from anyone running Ollama + GPU training setups. The local-first research loop is the use case I'm most excited about.
```
