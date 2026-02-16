# orze

A GPU experiment orchestrator that uses filesystem coordination. No databases, no message queues — just files.

Write your experiment ideas in a markdown file, and `farm.py` claims and runs them across your GPUs in parallel. It uses `mkdir` as an atomic lock, so it works across multiple machines with a shared filesystem.

Designed to be operated by LLM agents. See [RULES.md](RULES.md) for the complete LLM-readable specification.

**Website:** [orze.ai](https://orze.ai)

## How It Works

```
                 ideas.md                    results/
              ┌─────────────┐            ┌──────────────┐
              │ ## idea-001  │   claim    │ idea-001/    │
              │ ## idea-002  │──(mkdir)──>│   claim.json │
              │ ## idea-003  │            │   metrics.json│
              │ ...          │            │ idea-002/    │
              └─────────────┘            │   ...        │
                                         └──────────────┘
                     │                          ▲
                     ▼                          │
              ┌─────────────┐                   │
              │   farm.py   │───launch──────────┘
              │             │   (subprocess
              │  ┌───┐ ┌───┐│   per GPU)
              │  │GPU│ │GPU││
              │  │ 0 │ │ 1 ││
              │  └───┘ └───┘│
              └─────────────┘
```

The loop:
1. Parse `ideas.md` for experiment definitions
2. Find unclaimed ideas (no `results/{idea_id}/` directory)
3. Detect free GPUs (low memory usage, no active training)
4. **Claim** an idea by creating its results directory (`mkdir` — atomic, fails if exists)
5. **Launch** training as a subprocess with `CUDA_VISIBLE_DEVICES`
6. **Monitor** health — detect stalls, OOM, timeouts
7. **Evaluate** — run optional post-training eval script
8. **Report** — generate `results/report.md` leaderboard + `status.json`
9. Sleep, repeat

## Quick Start

```bash
# 1. Install dependencies
pip install torch torchvision pyyaml

# 2. Run one cycle (claims and trains the first unclaimed idea)
python farm.py --once

# 3. Check results
cat results/report.md
cat results/status.json

# 4. Run continuously on GPUs 0 and 1
python farm.py --gpus 0,1
```

## Project Configuration (orze.yaml)

For real projects, create an `orze.yaml` to configure paths, scripts, health monitoring, evaluation, and report format:

```bash
cp orze.yaml.example orze.yaml
# Edit orze.yaml for your project
python farm.py -c orze.yaml
```

See [orze.yaml.example](orze.yaml.example) for all options.

Key features enabled by orze.yaml:
- **Custom training script** with extra args and env vars
- **Post-training evaluation** (runs automatically after training succeeds)
- **Health monitoring** — stall detection, OOM detection, disk space checks
- **Failure handling** — auto-skip ideas after N failures, orphan cleanup
- **Custom report** — configurable columns, metrics from any JSON file

## The Protocol

### ideas.md format

Each idea is an H2 header with an embedded YAML config block:

~~~markdown
## idea-001: My Experiment Name
- **Priority**: high
- **Category**: architecture
- **Hypothesis**: Why this might work.

```yaml
model:
  type: simple_cnn
  channels: [32, 64, 128]
training:
  lr: 0.001
  epochs: 5
```
~~~

Priority controls execution order: `critical` > `high` > `medium` > `low`.

### Claiming (atomic mkdir)

When `farm.py` wants to run an idea, it calls `mkdir(results/idea-001/, exist_ok=False)`. On any POSIX filesystem, only one process can create a directory — the rest get `FileExistsError`. This is the entire coordination mechanism. It works across machines on NFS/EFS/FSx.

### metrics.json contract

Your training script must write `results/{idea_id}/metrics.json` when done:

```json
{
  "status": "COMPLETED",
  "test_accuracy": 0.9234,
  "test_loss": 0.2451,
  "training_time": 142.5
}
```

Status must be `"COMPLETED"` or `"FAILED"`. Add any other metrics you want — they'll show up in the report.

## Using Your Own Training Script

Replace `train_idea.py` with anything. The contract:

**Input:**
- `CUDA_VISIBLE_DEVICES` environment variable (set by farm.py)
- `--idea-id idea-001` — which idea to train
- `--results-dir results` — where to write output
- `--ideas-md ideas.md` — path to ideas file (read your config from here)
- `--config configs/base.yaml` — path to base config
- Plus any `train_extra_args` and `train_extra_env` from orze.yaml

**Output:**
- `results/{idea_id}/metrics.json` — must contain `{"status": "COMPLETED"|"FAILED", ...}`

**That's it.** Write metrics.json when done. Farm.py handles everything else.

## Multi-Machine Setup

If your machines share a filesystem (NFS, EFS, FSx, etc.):

```bash
# Machine 1
python farm.py -c orze.yaml --gpus 0,1,2,3

# Machine 2
python farm.py -c orze.yaml --gpus 0,1,2,3
```

Both instances read the same `ideas.md` and write to the same `results/` directory. The `mkdir` claim prevents duplicate work. Each machine's `claim.json` records which host claimed what.

## For LLM Agents

Read [RULES.md](RULES.md) — it contains the complete specification for operating orze:
- Ideas format and ID rules
- The metrics.json contract
- Experiment lifecycle (QUEUED → CLAIMED → TRAINING → COMPLETED → EVALUATED)
- How to read results, report.md, and status.json
- Configuration reference (orze.yaml)
- Best practices for writing experiment ideas

## CLI Reference

```
python farm.py [OPTIONS]

Options:
  -c, --config-file PATH   Path to orze.yaml project config
  --gpus GPU_IDS           Comma-separated GPU IDs (default: auto-detect)
  --timeout SECONDS        Max training time per idea (default: 3600)
  --poll SECONDS           Seconds between loop iterations (default: 30)
  --once                   Run one cycle and exit
  --report-only            Only regenerate report.md
  --ideas-md PATH          Path to ideas file (default: ideas.md)
  --base-config PATH       Path to base config (default: configs/base.yaml)
  --results-dir PATH       Results directory (default: results)
  --train-script PATH      Training script to run (default: train_idea.py)
  -v, --verbose            Debug logging
```

CLI args override orze.yaml values. Without orze.yaml, defaults match the original behavior.

## License

Apache 2.0
