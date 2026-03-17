# Orze — Development Guide

Read `SKILL.md` in this directory for the full operations & extension reference (CLI, API, architecture, diagnostics).

## Quick Reference

- **Run**: `orze -c orze.yaml` (auto-detect GPUs)
- **Stop**: `orze --stop`
- **Check**: `orze --check` (validate config)
- **Admin UI**: `orze --admin` → http://localhost:8787
- **MCP**: Claude Code connects via `.mcp.json` → `POST /mcp`

## Architecture (post-LOD refactor)

Every module in `src/orze/engine/` is <800 LOC with a calling spec at the top. Read the calling spec before reading the implementation.

```
engine/orchestrator.py  (657) — Orze class, run() recipe
engine/phases.py        (542) — main loop phases (mixin)
engine/role_runner.py   (499) — agent role lifecycle
engine/lifecycle.py     (390) — startup, shutdown, PID
engine/reporter.py      (329) — notifications + plateau
engine/upgrade.py       (223) — auto-upgrade pipeline
engine/cluster.py       (140) — multi-machine coordination
engine/config_dedup.py   (85) — config hash dedup
engine/retrospection.py  (67) — periodic analysis
skills/loader.py        (140) — composable prompt fragments
admin/mcp.py            (280) — MCP server for Claude Code
```

## Conventions

- LOD: max 800 LOC per file, calling spec at top, pure functions for tools
- Config: `orze.yaml` is the single source of truth
- Ideas: `ideas.md` is append-only, consumed into `idea_lake.db`
- Results: `results/{idea_id}/metrics.json` is the training contract
