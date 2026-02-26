# orze

Auto-research on autopilot. One script, one config, all GPUs.

Orze runs the full research loop: **generate ideas → train → evaluate → learn → repeat**. It coordinates GPUs via filesystem locks (`mkdir`), works across machines, and supports any LLM as the research agent — Claude, GPT, Gemini, local models, or your own script.

**Website:** [orze.ai](https://orze.ai)

## Install

```bash
curl -sL https://raw.githubusercontent.com/warlockee/orze/main/setup.sh | bash
```

This downloads the orze files into `orze/` under your project root. 

## Quick Start (3 minutes)

### 1. Create a minimal `train.py`
Your script receives `--idea-id`, `--config` (base), and `--ideas-md`. You are responsible for merging them.

```python
import argparse, json, yaml, os
from pathlib import Path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--idea-id", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--ideas-md", required=True)
    parser.add_argument("--results-dir", default="results")
    args = parser.parse_args()

    # 1. Load base config + idea overrides (Simplified)
    # For a production example, see orze/RULES.md
    with open(args.config) as f: config = yaml.safe_load(f)
    
    print(f"Training {args.idea_id} on GPU {os.environ.get('CUDA_VISIBLE_DEVICES')}...")
    
    # 2. Write results/idea-id/metrics.json when done
    res_dir = Path(args.results_dir) / args.idea_id
    res_dir.mkdir(parents=True, exist_ok=True)
    with open(res_dir / "metrics.json", "w") as f:
        json.dump({"status": "COMPLETED", "accuracy": 0.95}, f)

if __name__ == "__main__": main()
```

### 2. Configure and Launch
```bash
cp orze/orze.yaml.example orze.yaml
# Set train_script: train.py in orze.yaml

# Launch the orchestrator + self-healing watchdog
nohup python orze/farm.py -c orze.yaml >> results/farm.log 2>&1 &
nohup python orze/bug_fixer.py -c orze.yaml >> results/bug_fixer.log 2>&1 &
```

## Key Features

- **1M Scale Research** — Optimized SQLite-backed job queue and indexed reporting handles 1,000,000+ autonomous experiments with O(log N) scheduling.
- **Multi-LLM Research Army** — Run Claude, Gemini, GPT, and local models as parallel research agents. Orze auto-discovers API keys from your environment.
- **Delta Protocol** — Research agents only communicate configuration *changes*, reducing token costs by 60% and enabling massive context windows.
- **Systematic Safety** — Built-in Circuit Breaker stops the fleet if failure rates spike; Schema Validation catches hallucinations before they hit GPUs.
- **Self-Healing** — Companion `bug_fixer.py` watchdog auto-restarts crashed processes, kills stuck jobs, and diagnoses errors using an LLM.
- **Multi-Machine** — Orchestrate thousands of GPUs across different nodes via shared filesystems (NFS/EFS/FSx).
- **HP Sweep** — List-valued hyperparameters (e.g. `lr: [1e-4, 3e-4]`) auto-expand into Cartesian product sub-runs.
- **Cold Storage Archival** — Automatically moves bulky model checkpoints to cheap storage while keeping metadata on fast disks.
- **Admin Panel** — Real-time web dashboard at `:8787` for fleet monitoring and run management.
- **Telegram Bot** — Chat with your research cluster in natural language to check ranks or add new ideas.

## How It Works

```
┌─────────────────────────────────────────────────────┐
│                      farm.py                        │
│                                                     │
│   ┌───────────┐     ┌─────────┐     ┌──────────┐    │
│   │ Research  │────>│  Train  │────>│ Evaluate │    │
│   │ (any LLM) │     │ (GPUs)  │     │          │    │
│   └─────▲─────┘     └─────────┘     └──────────┘    │
│         │                                │          │
│         └────────── results/ ◄───────────┘          │
│                                                     │
│   ideas.md ◄── research ── report.md                │
└──────────────────────┬──────────────────────────────┘
                       │ monitors & heals
               ┌───────▼────────┐
               │  bug_fixer.py  │
               │   (watchdog)   │
               └────────────────┘
```

## The Contract

Your training script receives these standard arguments:
```bash
python train.py --idea-id idea-001 --results-dir results --ideas-md ideas.md --config base.yaml
```

**Required Output:** Your script must write `results/{idea_id}/metrics.json`:
```json
{"status": "COMPLETED", "test_accuracy": 0.92, "training_time": 142.5}
```

See [**RULES.md**](RULES.md) for the full technical specification.

## License

Apache 2.0
