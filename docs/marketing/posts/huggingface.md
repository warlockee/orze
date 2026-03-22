# HuggingFace Community Post / Space

## Rules of engagement
- HF community is ML-native. They know the tools, the workflow, the pain.
- Reference HF ecosystem (models, datasets, transformers) where relevant.
- Position as complementary to HF, not competing.
- They'll want to see how it integrates with their existing workflow.

---

## Blog Post (HuggingFace Blog or Community Post)

### Title
```
Orze: Autonomous GPU Experiment Orchestration for HuggingFace Model Training
```

### Body

```
## The Gap Between Training Runs

If you train models on GPUs — fine-tuning HuggingFace models, running ablations, exploring architectures — you know the workflow:

Launch training → wait → check results → decide what to try next → launch again.

The "decide what to try next" step is where GPU hours go to waste. Your H100 doesn't care that you're in a meeting. It just sits idle.

**Orze** is an open-source orchestrator that fills this gap. It keeps your GPUs busy running experiments continuously, and optionally uses LLM agents to generate new experiment ideas from your results.

## How It Works with HuggingFace

Your existing HF training script works as-is. Orze just needs it to save metrics to a JSON file:

```python
# train.py — your existing HuggingFace training script
from transformers import Trainer, TrainingArguments
import json, sys, yaml

# Orze passes the experiment config as the first argument
config = yaml.safe_load(open(sys.argv[1]))

training_args = TrainingArguments(
    output_dir=config.get("output_dir", "./output"),
    num_train_epochs=config.get("epochs", 3),
    learning_rate=config.get("learning_rate", 5e-5),
    per_device_train_batch_size=config.get("batch_size", 16),
    # ... your usual HF training args
)

trainer = Trainer(model=model, args=training_args, ...)
result = trainer.train()

# Save metrics for Orze
metrics = {
    "eval_accuracy": result.metrics.get("eval_accuracy", 0),
    "eval_loss": result.metrics.get("eval_loss", 0),
    "training_time": result.metrics.get("train_runtime", 0),
}
json.dump(metrics, open("metrics.json", "w"))
```

Then in `ideas.md`, define your experiments:

```markdown
## idea-bert-base: Fine-tune BERT-base
model_name: bert-base-uncased
learning_rate: 2e-5
epochs: 3
batch_size: 32

## idea-roberta: Fine-tune RoBERTa
model_name: roberta-base
learning_rate: 1e-5
epochs: 5
batch_size: 16

## idea-deberta: Fine-tune DeBERTa-v3
model_name: microsoft/deberta-v3-base
learning_rate: 3e-5
epochs: 3
batch_size: 16
```

Run `orze` and it trains all three in parallel across your GPUs.

## The LLM Research Agent

The real power comes from closing the loop. Configure a research agent:

```yaml
roles:
  research:
    mode: research
    backend: gemini
    model: gemini-2.5-pro
```

After your first batch of experiments completes, the agent reads the results:

- "BERT-base: 87.2% accuracy"
- "RoBERTa: 89.1% accuracy"
- "DeBERTa-v3: 90.4% accuracy"

And generates new ideas:

- "Try DeBERTa-v3 with larger batch size and learning rate warmup"
- "Ensemble DeBERTa-v3 + RoBERTa with learned weights"
- "Fine-tune DeBERTa-v3-large with gradient accumulation"
- "Add data augmentation with back-translation"

These get queued and trained automatically. The cycle repeats.

## Why Not Just Use Optuna?

Optuna optimizes hyperparameters for a fixed model. Orze (with the LLM agent) explores the *idea space*:

- Different model architectures (BERT vs RoBERTa vs DeBERTa)
- Different training strategies (standard fine-tuning vs LoRA vs prefix tuning)
- Different data strategies (augmentation, curriculum learning, hard negative mining)
- Different loss functions, schedulers, ensemble methods

The LLM agent can propose structurally different experiments, not just parameter variations.

## Quick Start

```bash
pip install orze
orze --init
# Edit orze.yaml and ideas.md
orze
```

- **GitHub**: https://github.com/warlockee/orze
- **License**: Apache 2.0
- **Requirements**: Python 3.9+, GPUs, that's it
```
