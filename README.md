# orze

Auto-research on autopilot. One script, one config, all GPUs.

Orze runs the full research loop: **generate ideas → train → evaluate → learn → repeat**. It coordinates GPUs via filesystem locks (`mkdir`), works across machines, and can use Claude as the research agent. No databases, no message queues — just files.

**Website:** [orze.ai](https://orze.ai)

## Install

```bash
curl -sL https://raw.githubusercontent.com/warlockee/orze/main/setup.sh | bash
```

This downloads 4 files into `orze/`. Then open [Claude Code](https://claude.ai/claude-code) and say:

```
@orze/AGENT.md set up and run experiments for this project
```

Claude will explore your codebase, create the config, write seed ideas, and launch the loop. That's it.

## How It Works

```
┌──────────────────────────────────────────────────┐
│                    farm.py                        │
│                                                   │
│   ┌─────────┐     ┌─────────┐     ┌──────────┐  │
│   │ Research │────>│  Train  │────>│ Evaluate │  │
│   │ (Claude) │     │ (GPUs)  │     │          │  │
│   └────▲────┘     └─────────┘     └──────────┘  │
│        │                                │         │
│        └────────── results/ ◄───────────┘         │
│                                                   │
│   ideas.md ◄── research ── report.md              │
└──────────────────────────────────────────────────┘
```

The loop:
1. **Research** — Claude (or any LLM) reads results, generates new experiment ideas
2. **Parse** `ideas.md` for experiment definitions
3. **Claim** unclaimed ideas via atomic `mkdir`
4. **Launch** training as subprocesses across free GPUs
5. **Monitor** health — stalls, OOM, timeouts, disk space
6. **Evaluate** — run post-training eval scripts
7. **Report** — update `results/report.md` leaderboard + `status.json`
8. Sleep, repeat

## Manual Setup (without Claude Code)

```bash
# 1. Install orze
curl -sL https://raw.githubusercontent.com/warlockee/orze/main/setup.sh | bash

# 2. Create your config (see orze/orze.yaml.example)
cp orze/orze.yaml.example orze.yaml
# Edit orze.yaml — set train_script, python path, report columns

# 3. Write seed ideas in ideas.md
# See orze/RULES.md for the exact format

# 4. Run
python orze/farm.py -c orze.yaml
```

## Key Features

- **Agent roles** — multiple agents (research, documenter, analyzer, etc.) run alongside training
- **Research agent** — Claude CLI or any script generates ideas automatically
- **Multi-GPU** — claims and trains across all GPUs in parallel
- **Health monitoring** — stall detection, OOM detection, disk space checks
- **Post-training eval** — runs automatically after each successful training
- **Configurable report** — custom columns, metrics from any JSON file
- **Multi-machine** — works across machines on shared filesystems (NFS/EFS/FSx)
- **Failure handling** — auto-skip after N failures, orphan cleanup
- **Atomic coordination** — `mkdir` as lock, no race conditions

## The Contract

Your training script receives these args from orze:

```
CUDA_VISIBLE_DEVICES=N python train.py --idea-id idea-001 --results-dir results --ideas-md ideas.md --config base.yaml
```

Your script must write `results/{idea_id}/metrics.json`:

```json
{"status": "COMPLETED", "test_accuracy": 0.92, "training_time": 142.5}
```

That's it. See [RULES.md](RULES.md) for the full specification.

## Documentation

| File | For |
|------|-----|
| [AGENT.md](AGENT.md) | Claude Code bootstrap — `@orze/AGENT.md` |
| [RULES.md](RULES.md) | Complete LLM-readable specification |
| [orze.yaml.example](orze.yaml.example) | All configuration options |

## CLI Reference

```
python orze/farm.py [OPTIONS]

  -c, --config-file PATH   Path to orze.yaml
  --gpus GPU_IDS           Comma-separated GPU IDs (default: all)
  --once                   Run one cycle and exit
  --report-only            Only regenerate report.md
  --role-only NAME         Run a single agent role once and exit
  --research-only          Alias for --role-only research
  -v, --verbose            Debug logging
```

## License

Apache 2.0
