---
name: research
---

## Best Practices for Writing Ideas

1. **Start with baselines** — simple models first, then iterate
2. **Vary one thing at a time** — easier to attribute improvements
3. **Use priority** — mark promising directions as `high`, speculative as `low`
4. **Include hypotheses** — helps interpret results and plan next ideas
5. **Check the leaderboard** before generating similar ideas — avoid redundant work
6. **Use categories** — group related ideas for easier analysis
7. **Track lineage** — set `Parent: idea-XXX` to trace what inspired each idea
8. **Keep YAML configs complete** — don't rely on implicit defaults
9. **React to results** — combine winners, diagnose failures, push best approaches further
10. **Span diverse categories** — architecture, training, data, loss, ensemble — not just hyperparameter tweaks

## Reading Results

### results/report.md
Auto-generated leaderboard. Columns are configurable via `orze.yaml`. Sorted by primary metric.

### results/status.json
Machine-readable status, updated every iteration:

```json
{
  "timestamp": "2026-02-16T14:30:00",
  "iteration": 142,
  "active": [{"idea_id": "idea-045", "gpu": 3, "elapsed_min": 12.5}],
  "free_gpus": [0, 1, 2, 4, 5, 6, 7],
  "queue_depth": 87,
  "completed": 55,
  "failed": 3,
  "skipped": 2,
  "disk_free_gb": 1024.5,
  "top_results": [...]
}
```

Use this to monitor progress programmatically.

### Per-Idea Files
Each `results/{idea_id}/` contains:
- `claim.json` — who claimed it, when, on which GPU
- `train_output.log` — stdout/stderr from training
- `metrics.json` — final metrics (the contract)
- `eval_output.log` — eval stdout (if eval configured)
- Other files written by the training/eval/post scripts

## Phase Transitions

For projects with distinct research phases, use marker files:

- **Phase 1 (Build)**: Infrastructure setup, smoke test. Create `.phase1_complete` when done.
- **Phase 2 (Explore)**: Broad exploration. Orze runs ideas across all GPUs.
- **Phase 3 (Converge)**: Focus on top approaches. Create `.phase3_started`.

Orze itself doesn't enforce phases — it just runs whatever ideas are in ideas.md. The research agent should check phase markers and adjust its idea generation strategy accordingly. For example, in Phase 3, generate only ideas that build on the top 3 approaches.
