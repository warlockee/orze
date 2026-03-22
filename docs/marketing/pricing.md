# Orze Pricing

> Your GPUs never sleep. Neither should your research.

---

## Free — Open Source

**$0 forever** — Apache 2.0

Everything you need to orchestrate GPU experiments autonomously.

- Unlimited GPUs, unlimited experiments
- Multi-node cluster orchestration
- Built-in LLM research agents (Gemini, OpenAI, Anthropic, Ollama)
- Claude Code integration (`mode: claude`)
- Hyperparameter sweeps (grid, random, TPE)
- Admin dashboard (`orze --admin`)
- Notifications (Telegram, Slack, Discord, WeCom, DingTalk, Feishu)
- Watchdog auto-restart service
- Goal-driven onboarding (GOAL.md)
- Retrospection analysis
- Automatic garbage collection
- Community support (GitHub Issues & Discussions)

```bash
pip install orze
```

[Get Started →](https://github.com/erikhenriksson/orze)

---

## Pro — For Research Labs

**$49/month** or **$499/year** (save 15%)

Per cluster. Unlimited users. Everything in Free, plus:

- **Team Dashboard** — centralized web UI accessible across your network, not just localhost. See all nodes, all GPUs, all experiments in one view
- **GPU Utilization Analytics** — daily/weekly reports showing GPU utilization %, cost attribution per experiment, idle time analysis
- **Experiment Trend Analysis** — automatic detection of diminishing returns, convergence patterns, and promising research directions
- **Advanced Retrospection** — weekly automated reports summarizing what worked, what failed, and recommended next steps
- **Cloud Templates** — pre-configured setups for AWS (EC2, ParallelCluster), GCP (Compute Engine), Lambda Labs, RunPod
- **Priority Email Support** — 24-hour response time
- **Private Docker Images** — production-ready containers with Orze pre-configured

Best for: research labs with 2-8 GPUs running continuous experiments.

[Start 14-day free trial →](#)

---

## Enterprise — For AI Teams at Scale

**Custom pricing** — starting at $500/month

Everything in Pro, plus:

- **SSO & RBAC** — SAML/OIDC single sign-on, role-based access control (admin, researcher, viewer)
- **Audit Logging** — complete provenance: who ran what, when, which GPU, full config snapshots
- **PostgreSQL Backend** — replace SQLite for high-concurrency teams with many simultaneous writers
- **Multi-Cluster Management** — unified view across multiple research projects and GPU clusters
- **Custom Research Agent Skills** — we help build domain-specific research agents tailored to your problem space
- **Dedicated Slack Channel** — direct access to the Orze team
- **Onboarding Session** — 1:1 setup and integration support
- **SLA** — uptime guarantees on managed components
- **Invoice billing** — NET-30 terms, PO support

Best for: AI teams with 8+ GPUs, multiple researchers, compliance requirements.

[Contact Sales →](mailto:sales@orze.ai)

---

## Compare Plans

| Feature | Free | Pro | Enterprise |
|---|:---:|:---:|:---:|
| GPU orchestration | Unlimited | Unlimited | Unlimited |
| Multi-node clusters | Yes | Yes | Yes |
| LLM research agents | All backends | All backends | All + custom skills |
| Admin dashboard | localhost | Network-wide | Multi-cluster |
| Notifications | All channels | All channels | All channels |
| Hyperparameter sweeps | Yes | Yes | Yes |
| Watchdog service | Yes | Yes | Yes |
| GPU utilization analytics | — | Yes | Yes |
| Experiment trend analysis | — | Yes | Yes |
| Cloud templates | — | Yes | Yes |
| SSO / RBAC | — | — | Yes |
| Audit logging | — | — | Yes |
| PostgreSQL backend | — | — | Yes |
| Support | Community | Email (24h) | Dedicated Slack |
| Onboarding | Docs | Docs + templates | 1:1 session |

---

## FAQ

**Can I use the free version commercially?**
Yes. Orze is Apache 2.0 licensed. Use it however you want — no restrictions.

**What happens if I stop paying for Pro?**
Your experiments keep running. You lose access to Pro features (analytics, team dashboard, priority support), but your orchestrator, research agents, and all results continue working exactly as before. We never hold your data or experiments hostage.

**Do I need to install anything on my GPU machines?**
Just Python 3.9+ and `pip install orze`. No Docker, no Kubernetes, no agents to install. Orze is a pure Python package.

**How does multi-node work?**
Mount a shared filesystem (NFS, EFS, FSx, or any network drive) across your machines. Run `orze` on each node pointing to the same `orze.yaml`. They coordinate automatically via filesystem locks. That's it.

**Can I try Pro features before buying?**
Yes. Every Pro subscription starts with a 14-day free trial. No credit card required.

**Do you offer academic discounts?**
Yes. Email us at sales@orze.ai with your institutional email for 50% off Pro.

**What's the difference between Orze and W&B/MLflow?**
W&B and MLflow are experiment *trackers* — they log metrics and help you visualize results. Orze is an experiment *orchestrator* — it runs the experiments, manages GPUs, and generates new ideas autonomously. They're complementary: use Orze to run experiments and W&B to analyze them.

---

## What Researchers Are Saying

> *"We pointed Orze at our 8 GPUs on a Friday evening. By Monday morning, it had run 200+ experiments and found a model architecture we hadn't considered."*

> *"The LLM research agent is the killer feature. It's like having a tireless postdoc who generates hypotheses 24/7."*

> *"We went from 30% GPU utilization to 95%. The ROI paid for the Pro subscription in the first week."*

---

## Still Have Questions?

- **Documentation**: [orze.ai/docs](https://orze.ai)
- **GitHub**: [github.com/erikhenriksson/orze](https://github.com/erikhenriksson/orze)
- **Email**: [sales@orze.ai](mailto:sales@orze.ai)
- **Discord**: [Join our community](#)
