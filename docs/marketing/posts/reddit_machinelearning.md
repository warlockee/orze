# Reddit r/MachineLearning — [P] Project Post

## Rules of engagement
- r/ML is academic. They want methodology, not hype.
- Use the [P] tag (Project).
- Be honest about limitations. This community will call out BS instantly.
- Include technical depth. They'll respect the engineering.
- Don't be salesy. Frame it as sharing work, not promoting a product.

---

## Title

```
[P] Orze: Open-source GPU experiment orchestrator with autonomous LLM research agents — ran 800+ experiments for Kaggle competition
```

## Body

```
Hi r/MachineLearning,

I've been working on Orze, an open-source tool for automating the full ML experiment loop. I wanted to share it because I think the "LLM agent as research idea generator" approach is underexplored as practical infrastructure (vs. one-off demos like AI Scientist).

**What it does**

Orze is a GPU experiment orchestrator. You give it a training script, a config file, and GPUs. It:

1. Auto-detects available GPUs
2. Picks unclaimed experiments from a queue (ideas.md — a Markdown file)
3. Runs training, collects metrics
4. Updates a leaderboard
5. Optionally: an LLM research agent reads results and generates new experiment ideas, closing the loop

The key insight is that the bottleneck in most research workflows isn't training time — it's the latency between "results came in" and "I decided what to try next." An LLM agent can reduce that to zero.

**Architecture choices**

- Filesystem-based coordination (no K8s, no Redis, no message queue)
- Ideas stored as Markdown, results as JSON
- Multi-node via shared filesystem + filesystem locks
- SQLite for historical analysis (idea_lake.db)
- Pure Python, `pip install orze`

I deliberately avoided heavyweight infrastructure. The target is "researcher with SSH access to a GPU machine," not "DevOps team with a Kubernetes cluster."

**LLM research agent**

The built-in research agent supports multiple backends (Gemini, OpenAI, Anthropic, local Ollama, any OpenAI-compatible endpoint). It:

- Reads the full leaderboard and completed experiment configs
- Analyzes patterns (what architectural choices correlate with good results)
- Generates structurally diverse experiment ideas (not just HP variations)
- Appends them to the experiment queue
- Optionally uses a retrospection analysis (periodic summary of what's working)

This is different from Optuna/Ray Tune, which optimize parameters for a fixed model. The LLM agent can propose entirely different architectures, loss functions, or data preprocessing strategies.

**Kaggle case study**

I used Orze for the Nexar Dashcam Collision Prediction competition. Setup: 8 H100s, Gemini 2.5 Pro as the research agent, running continuously over a weekend.

Results:
- 800+ experiments completed autonomously
- The LLM agent discovered "time-to-event matched training" — a strategy where training samples are weighted by temporal proximity to the collision event. This emerged from the agent analyzing failure patterns across hundreds of experiments.
- Achieved local mAP of 0.907 (competitive with winning approaches)

The time-to-event insight was not something I would have tried. The agent noticed that models consistently struggled with videos where the collision happened late in the clip, and proposed a fix.

**Limitations (honest assessment)**

- The LLM agent generates plenty of bad ideas. The signal-to-noise ratio improves over time as it has more data to learn from, but early cycles can be noisy.
- Filesystem-based coordination has scalability limits. Fine for 1-4 nodes, gets slow beyond that.
- No distributed training support — Orze orchestrates single-GPU experiments. If you need DDP, use your training script's built-in support.
- The research agent works best when the search space is large (many possible architectures/approaches). For narrow HP tuning, Optuna is better.

**Links**

- GitHub: https://github.com/warlockee/orze
- PyPI: `pip install orze`
- License: Apache 2.0

Happy to discuss the architecture, the LLM agent design, or the Kaggle experience. I'm particularly interested in hearing from people who have tried similar "LLM-in-the-loop" research automation approaches.
```
