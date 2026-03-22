# Orze Go-to-Market Strategy

## Positioning

> "Orze is the autopilot for GPU experiments. Point it at your GPUs and a research question — it generates hypotheses, runs experiments, learns from results, and repeats. No Kubernetes. No cloud lock-in. One YAML file."

Orze is NOT an experiment tracker (W&B, MLflow, Neptune). It is an autonomous experiment **orchestrator** with a closed loop. Different category entirely.

---

## Competitive Landscape

| Competitor | What they do | How Orze differs |
|---|---|---|
| W&B / MLflow / Neptune | Log metrics, visualize, compare runs | Orze *runs* experiments, not just tracks them |
| NVIDIA Run:ai | GPU cluster orchestration (enterprise) | Orze is lightweight, no K8s needed |
| Sakana AI Scientist | Autonomous research paper generation | Orze is practical infrastructure, not a demo |
| Agent Laboratory | LLM-driven research agents | Orze integrates orchestration + execution |
| Kubeflow / SageMaker | Heavy MLOps platforms | Orze is zero-config, no vendor lock-in |

---

## Target Customers

### Tier 1: Individual Researchers & PhDs (Free)
- **Pain**: manually queuing experiments, wasting GPU idle time
- **Value**: "Your GPUs run experiments 24/7 while you sleep"
- **Acquisition**: organic (GitHub, PyPI, Twitter/X, Reddit, HuggingFace)

### Tier 2: Small Research Labs (3-20 people) — Primary Revenue
- **Pain**: coordinating GPU access, no systematic idea exploration
- **Value**: "Multi-node orchestration + LLM agents = 10x more experiments per GPU-week"
- **Acquisition**: content marketing, conferences, word-of-mouth

### Tier 3: Enterprise AI Teams (50+) — Future Revenue
- **Pain**: GPU utilization below 40%, manual management, compliance
- **Value**: "Maximize GPU ROI with autonomous orchestration"
- **Acquisition**: outbound sales, cloud provider partnerships

---

## Pricing (Open Core)

| Tier | Price | Target |
|---|---|---|
| Free (OSS) | $0 | Individuals, PhDs |
| Pro | $49/mo or $499/yr | Small labs (2-8 GPUs) |
| Enterprise | $500-2K/mo | AI teams (8+ GPUs) |

See [pricing.md](pricing.md) for full pricing page copy.
See [pro_features_spec.md](pro_features_spec.md) for Pro feature specifications.

---

## Distribution Channels

### 1. Content Marketing (Highest Priority)

**Week 1-2 actions:**
- Write launch blog post → [launch_blog_post.md](launch_blog_post.md)
- Record 3-minute demo video: `pip install orze` → first autonomous experiment completing
- Post to YouTube, embed on orze.ai

**Week 3-4 actions:**
- Post to Hacker News: "Show HN: Orze — GPU experiment orchestrator with autonomous LLM research agents"
- Post to Reddit: r/MachineLearning, r/LocalLLaMA, r/selfhosted
- Post to Twitter/X with demo video
- Submit to Product Hunt

**Week 5-8 actions:**
- Publish case study: "How Orze Ran 800 Experiments Autonomously for a Kaggle Competition"
- Create "Getting Started in 5 Minutes" tutorial
- Engage in ML communities (answer GPU orchestration questions, mention Orze where relevant)

### 2. GitHub Optimization

- Fix repo discoverability (consistent URL, topics, description)
- Add topics: `gpu`, `experiment-orchestration`, `ml-automation`, `research-agent`, `llm`
- Create GitHub Discussions
- Add CONTRIBUTING.md
- Target: 100 stars in first month

### 3. PyPI Optimization

- Rewrite description (currently just "orze.ai")
- New description: "Autonomous GPU experiment orchestrator with LLM research agents. Generate ideas, train, evaluate, learn, repeat."
- Publish changelog on each release

### 4. Conference Presence

- Submit talk/poster to NeurIPS 2026 or ICML 2026 workshops
- Title: "Autonomous ML Research via LLM-Orchestrated Experiment Loops"
- Present at ML meetups (virtual counts)

### 5. Strategic Integrations

- HuggingFace: one-click setup for HF model training
- W&B: position as complementary ("Orze runs, W&B tracks")
- Cloud templates: AWS, GCP, Lambda Labs

---

## Messaging Framework

### Tagline Options
1. "The autopilot for GPU experiments"
2. "Your GPUs never sleep"
3. "From hypothesis to result, automatically"

### Elevator Pitch (30 seconds)
> "Orze is an open-source GPU experiment orchestrator that runs the full ML research loop autonomously. You give it a research question and access to GPUs. It uses LLM agents to generate experiment ideas, trains them, evaluates results, learns what works, and generates better ideas. One YAML file, no Kubernetes, scales from one GPU to hundreds across machines."

### By Audience

**Researchers**: "Stop babysitting experiments. Orze explores the hypothesis space while you think about the big picture."

**Lab managers**: "Your 8 GPUs are idle 60% of the time. Orze keeps them running 24/7 with intelligent experiment scheduling."

**Enterprise**: "Maximize GPU ROI with autonomous experiment orchestration. No infrastructure changes needed."

---

## 90-Day Action Plan

### Week 1-2: Foundation
- [ ] Fix GitHub repo discoverability
- [ ] Rewrite PyPI package description
- [ ] Record 3-minute demo video
- [ ] Set up Discord community server
- [ ] Write launch blog post

### Week 3-4: Launch Push
- [ ] Post to Hacker News (Show HN)
- [ ] Post to Reddit (r/MachineLearning, r/LocalLLaMA, r/selfhosted)
- [ ] Post to Twitter/X with demo video
- [ ] Submit to Product Hunt
- [ ] Email 10 ML researchers personally

### Week 5-8: Content Engine
- [ ] Publish Kaggle competition case study
- [ ] Create "Getting Started in 5 Minutes" tutorial
- [ ] Record 15-20 min walkthrough video
- [ ] Begin community engagement
- [ ] Build email list on orze.ai

### Week 9-12: Monetization Prep
- [ ] Build Pro tier MVP (team dashboard + analytics)
- [ ] Set up Stripe/Lemon Squeezy billing
- [ ] Create pricing page on orze.ai
- [ ] Reach out to 5 labs for Pro beta testing
- [ ] Submit NeurIPS/ICML workshop paper

---

## Revenue Projections (Conservative)

| Month | Free Users | Pro ($49/mo) | Enterprise | MRR |
|---|---|---|---|---|
| 3 | 100 | 0 | 0 | $0 |
| 6 | 500 | 15 | 0 | $735 |
| 9 | 1,500 | 40 | 1 | $2,960 |
| 12 | 3,000 | 80 | 3 | $7,420 |

---

## Key Risks

| Risk | Mitigation |
|---|---|
| "Why not a bash script?" | Emphasize closed-loop LLM agents. Bash doesn't generate hypotheses. |
| Large players build this | Move fast on community. Lightweight = permanent advantage. |
| AI Scientist eats our lunch | Those are demos, not production tools. We are infrastructure. |
| Nobody pays for OSS | Pro must offer genuine time savings, not just gated features. |

---

## The One Thing to Do First

**Write the launch blog post and post it to Hacker News.**

Your story is compelling: solo developer, LLM research agents, GPU clusters, Kaggle competition, open source. HN loves this. A strong Show HN can generate 50-200 stars in a day.

Distribution before monetization. Everything else depends on having users first.
