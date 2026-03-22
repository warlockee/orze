# Reddit r/selfhosted Post

## Rules of engagement
- This community values: no cloud dependency, lightweight, easy to deploy, no vendor lock-in
- Lead with the self-hosted angle — no external services required
- Mention: no Docker required, no K8s, pure Python, filesystem-based
- Keep it practical — what does it do, how do I run it

---

## Title

```
Orze — self-hosted GPU experiment orchestrator, no Docker/K8s needed, pure Python, filesystem-based coordination
```

## Body

```
I built Orze for automating ML experiments across GPUs. Sharing here because I think the self-hosted crowd will appreciate the architecture: **no Docker, no Kubernetes, no database server, no message queue, no cloud dependencies.**

**What it is**

A GPU experiment orchestrator. You give it a training script and a list of experiments. It auto-detects your GPUs, keeps them all busy running experiments, collects results, and maintains a leaderboard.

Optional: add an LLM agent (local via Ollama, or any API) that reads results and generates new experiment ideas automatically.

**Stack**

- Pure Python package (`pip install orze`)
- Config: single YAML file
- Experiment queue: Markdown file (human-readable, git-friendly)
- Results: JSON files (one directory per experiment)
- Coordination: filesystem locks (works on NFS, EFS, any shared mount)
- History: SQLite (auto-managed, no server)
- Admin UI: built-in FastAPI server (`orze --admin`)

**Multi-node**

Mount a shared filesystem across machines. Run `orze` on each node. They coordinate automatically via filesystem locks. That's the entire setup.

No service mesh, no etcd, no ZooKeeper. Just NFS + Python.

**Deployment**

```bash
pip install orze
orze --init          # generates config + scaffolding
orze                 # starts
orze service install # optional: systemd/cron watchdog
```

**Notifications**

Supports Telegram, Slack, Discord, WeCom, DingTalk, Feishu, and generic webhooks. All optional, configured in YAML.

**Resource usage**

The orchestrator itself is lightweight — a single Python process that polls every 30 seconds. The actual GPU work is done by your training scripts (spawned as subprocesses). Minimal overhead.

**Links**

- GitHub: https://github.com/erikhenriksson/orze
- License: Apache 2.0
- No telemetry, no phone-home, no accounts required

It's primarily aimed at ML researchers but the orchestration pattern (queue of tasks → distribute across GPUs → collect results → generate new tasks) might be useful for other GPU workloads too.
```
