# orze

Agentic Auto-research on autopilot. One package, one config, all GPUs.

Orze runs the full research loop: **generate ideas → train → evaluate → learn → repeat**. It coordinates GPUs via filesystem locks (`mkdir`), works across machines, and supports any LLM as the research agent — Claude, GPT, Gemini, local models, or your own script.

**Website:** [orze.ai](https://orze.ai)

## Install

```bash
pip install orze
```

## Quick Start

```bash
# If you in Claude/Gemini/Codex Cli
do @orze/AGENT.md

---- If you want the old school way ----
# Initialize a new project (creates train.py, orze.yaml, ideas.md)
orze --init

# Launch the orchestrator
orze
```

That's it. Orze will auto-detect your GPUs and start running experiments from `ideas.md`.

## Key Features

- **Scales to 1M+ Experiments** — SQLite-backed job queue and indexed reporting with O(log N) scheduling.
- **Multi-LLM Research Army** — Run Claude, Gemini, GPT, and local models as parallel research agents. Auto-discovers API keys from your environment.
- **Delta Protocol** — Research agents only communicate configuration *changes*, reducing token costs by 60%.
- **Circuit Breaker** — Stops the fleet if failure rates spike. Schema validation catches hallucinations before they hit GPUs.
- **Self-Healing Watchdog** — Companion `bug_fixer` agent auto-restarts crashed processes, kills stuck jobs, and diagnoses errors using an LLM.
- **Multi-Machine** — Orchestrate thousands of GPUs across nodes via shared filesystems (NFS/EFS/FSx).
- **HP Sweep** — List-valued hyperparameters (e.g. `lr: [1e-4, 3e-4]`) auto-expand into Cartesian product sub-runs.
- **Admin Panel** — Real-time web dashboard at `:8787` for fleet monitoring. Install with `pip install orze[admin]`.
- **Clean Uninstall** — `orze --uninstall` removes all runtime files and the package itself, preserving only your research results.

## How It Works

```
┌─────────────────────────────────────────────────────┐
│                       orze                          │
│                                                     │
│   ┌───────────┐     ┌─────────┐     ┌──────────┐   │
│   │ Research  │────>│  Train  │────>│ Evaluate │   │
│   │ (any LLM) │     │ (GPUs)  │     │          │   │
│   └─────▲─────┘     └─────────┘     └──────────┘   │
│         │                                │          │
│         └────────── results/ ◄───────────┘          │
│                                                     │
│   ideas.md ◄── research ── report.md                │
└─────────────────────────────────────────────────────┘
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

See [**RULES.md**](RULES.md) for the full technical specification.

## License

Apache 2.0
