# Changelog

## Unreleased

### Changed
- **`.orze/` directory layout refactor** — Runtime state moved from `results/` to `.orze/` for cleaner git workflows
  - `results/` → `orze_results/` (renamed to avoid `.gitignore` collisions)
  - `results/_research_logs/` → `.orze/logs/`
  - `results/_receipts/` → `.orze/receipts/`
  - `results/idea_lake.db` → `.orze/idea_lake.db`
  - `results/ideas.md` → `.orze/ideas.md`
  - Per-role rules (e.g., `ENGINEER_RULES.md`) → `.orze/rules/`
  - See README "File Layout" section for full mapping

### Added
- `orze admin migrate` — One-shot migration command for existing projects (auto-runs on first `orze run` if needed)
- `orze upgrade` — One-liner to reinstall orze + orze-pro from source and restart daemon if running
- `orze init` now creates `.orze/` structure with `version.json` for forward compatibility
- `needs_intervention` notification event — Alerts for blocked conditions (HF gated models, missing API keys, disk full, OOM, etc.)
- Stray file sweeper — Quarantines role-generated files at project root to `orze_results/stray/` (configurable via `sweep_stray: true`)
- Intervention detection patterns for common blockers (HuggingFace gated models, missing tokens, GitHub auth, OpenAI/Anthropic keys, disk full, CUDA OOM, sudo prompts)

### Fixed
- Hardcoded `"results"` paths throughout codebase replaced with `orze_path(cfg, kind, name)` helper
- Default `ideas_file` and `idea_lake_db` now resolve correctly to `.orze/` location


## 3.7.0 — Honest-eval + cleanup

Fixes the one remaining fiction in the v3.6.2 self-evolving proof: the
0.9057 headline was computed with a 5-fold CV-OOF on **test** labels,
so the α that maximizes the reported metric was fit on the exact rows
it was then reported against — a classic label-peek. v3.7.0 removes
that leakage from the promotion path.

### Breaking

- **CV-OOF on test labels no longer eligible for promotion.** The old
  `cv_mix` calibrator remains in the registry for reference-only
  reproduction, but any `metrics.json` that declares `honest: false`
  (or that was produced by the legacy path) is refused by
  `champion_guard.check_promotion`. Promote-flow downstream code that
  used to read `pgmAP_ALL` uncritically must now honor the
  `honest` flag.

### Added

- **F11 `cv_mix_honest` calibrator** — explicit `fit(fit_mask, …)` /
  `apply(…)` contract. Hyperparameters (α) are selected only on
  `fit_mask` rows; apply is mask-free (works on the full population
  with the chosen α). Also adds `per_group_rank_normalize()` as a
  public helper so adapters can assemble parts without peeking.
- **Split-aware `bundle_combiner.search`.** New keyword
  `splits={"val": mask, "report": {name: mask|None, …}}`. Tuning uses
  `val`; each `report` mask is scored separately. Leak guard: if
  every report mask is identical to `val`, raises `ValueError`.
  Legacy single-mask behavior preserved when `splits` is `None`
  (returns `honest: false`).
- **Honest-flag enforcement in `champion_guard`.** `metrics.json` with
  `honest: false` cannot be promoted.
- **`posthoc_defaults` in `orze.yaml`.** Top-level dict (e.g.
  `solution_csv`, `project_root`, `python`) is merged into every
  posthoc idea before dispatch — keeps per-idea config thin and
  avoids hand-stamping infra paths into `search_role` output.
- **Nexar-collision adapter is now F11-native.** Reads per-frame NPZs
  directly, uses `per_group_rank_normalize` + `cv_mix_honest`, and
  writes `metrics.json` with `pgmAP_Public`, `pgmAP_Private`,
  `pgmAP_ALL_honest`, and `alpha_tuned_on_public`.
- **`tests/test_honest_eval.py`** — `fit_mask` respect, empty-mask
  error, split-aware reporting, leak detection, adapter end-to-end,
  champion_guard honest gating.

### Changed (refactor)

- The Nexar-collision adapter **no longer imports
  `nexar_collision/scripts/eval_tta_npz.py`**. The consumer script is
  kept in-repo only for the manual reproducer (which is not on the
  promotion path). This closes the last adapter↔consumer coupling.

### Chore

- `orze/docs/audit_v3_6_2.md` — matrix proving every F8..F16 module is
  either live-exercised or test-covered (no dead code).
- Consumer-side: 36 superseded `eval_*.py` scripts archived to
  `nexar_collision/_archive/eval_superseded/`. The live set is
  `scripts/eval_tta_npz.py`, `scripts/eval_champion_0905_final.py`,
  `eval_agg_sweep.py`, `eval_e2e.py`. `ideas.md.safe*` backups and
  the `_corrupt_ideas/` directory removed.

### Verified numbers (live, autonomous)

| metric               | value  |
| -------------------- | -----: |
| `pgmAP_Public`       | 0.9209 |
| `pgmAP_Private`      | 0.8928 |
| `pgmAP_ALL_honest`   | 0.9060 |
| `alpha_tuned_on_public` | 0.85 |

See `nexar_collision/results/_orze_honest_proof.md` for the
reproducer and full acceptance table.


## 3.6.0 — Post-hoc search capability

Turns the manual Nexar Collision 0.8994 → 0.9057 recovery into a
first-class orze capability: post-hoc inference search (agg × calib ×
TTA bundles) is now a native scheduling path, complete with a
single-model bundle discipline, a catalog of inference artifacts, a
promotion guard, and a tight-loop search role.

### Added

- **F8 — idea.kind field.** `ideas` schema gains a `kind` column
  (`train` | `posthoc_eval` | `tta_sweep` | `agg_search` |
  `bundle_combine` | `audit`). Any other value is a hard error.
  YAML configs accept `kind:`. Back-compat migration via
  `idea_lake._migrate_if_needed` — existing rows default to `train`.
- **F9 — ArtifactCatalog.** New SQLite-backed catalog at
  `results/idea_lake_artifacts.db` tracking ckpts, preds_npz,
  features, and tta_preds keyed by `ckpt_sha` (SHA-256 of first +
  last 4 MB of the weight file). `orze catalog scan` walks
  `results/`; `metric_harvester` auto-registers artifacts produced
  by new experiments.
- **F10 — InferenceBundle + bundle_combiner.** An
  `InferenceBundle` is ≥ 2 `preds_npz` artifacts with the SAME
  `ckpt_sha` (TTA views of ONE model, not an ensemble). The combiner
  sweeps k × subsets × aggregations × calibrators and records the
  winner as a `kind='bundle_combine'` idea. Mixed-sha loads raise
  `ValueError` — single-model discipline is enforced.
- **F11 — Aggregation + calibration registry.** A single `REGISTRY`
  dict of aggregations (`last`, `mean`, `max`, `late_k1..k5`,
  `top_k_mean(k)`, `exp_weighted(alpha)`, `noisy_or`,
  `dense_late_softmax(t)`) and calibrators (`identity`, `platt`,
  `isotonic`, `group_calibrated`, `cv_mix`). `agg_search` does a
  nested-CV sweep over the registry and records the winner as a
  `kind='agg_search'` idea. No test-set leakage by construction.
- **F12 — posthoc_runner + launcher kind dispatch.** New inference-
  only runner (`src/orze/engine/posthoc_runner.py`) with pluggable
  adapters (`@register_adapter`). Built-in `null` adapter (canned
  metrics, used in tests) and `nexar_collision` adapter (shells out
  to `eval_tta.py` / `eval_champion_0905_final.py` /
  `eval_agg_sweep.py` without modifying consumer training scripts).
  `launcher.launch()` now dispatches non-`train` ideas through this
  path; the `train` path is preserved byte-exact.
- **F13 — Cross-host opportunistic scheduler.** `gpu_slots.poll_fleet`
  / `pick_least_loaded` / `wrap_remote_cmd`. Cheap ideas
  (`posthoc_eval`, `tta_sweep`, `bundle_combine`, `agg_search`) can
  be dispatched to the least-loaded GPU across all `hosts:` listed in
  `orze.yaml` (ssh `BatchMode=yes`). Training stays host-pinned.
  Only the F2 leader assigns remote work.
- **F14 — Champion-promotion guard.** Before firing `new_best`,
  `champion_guard.check_promotion` (a) re-verifies the claim
  (rerunning `idea.reproducer` if present, else reading
  `metrics.json`) and (b) computes z-score against the rolling
  last-50 promotion distribution. `z > 4.0` blocks the promotion,
  fires Telegram `notify('audit', …)`, and creates a
  `kind='audit'` follow-up idea that flows into `bug_fixer`.
  Config keys in `orze.yaml`:
  `champion_guard: {enabled, z_threshold, min_history}`.
- **F15 — Tight-loop `search` role.** Default `cooldown: 30`,
  `timeout: 600`. Composes agg_search → bundle_combiner → posthoc_
  runner: reads the ArtifactCatalog, enqueues `agg_search` and
  `bundle_combine` ideas until no free combinations remain or the
  champion hasn't moved in N cycles. Idempotent. Opt-in via
  `orze.yaml.example`.
- **F16 — Retroactive champion ingest.** `orze ingest-champion
  --results-dir … --idea-id …` (and `python -m
  orze.agents.ingest_champion`) records a pre-v3.6.0 manual champion
  as a `kind='bundle_combine'` idea, registers its ckpt + the 4 TTA
  NPZs in the ArtifactCatalog under the same ckpt_sha, and updates
  `_orze_state.json:best_idea_id`. Used to preserve the Nexar
  Collision 0.9057 (Public 0.9199 / Private 0.8929) record after the
  upgrade.

### Tests

50 new tests (177 total, all green). Dedicated suites:
`test_idea_kind.py`, `test_artifact_catalog.py`, `test_aggregations.py`,
`test_agg_search.py`, `test_posthoc_runner.py`,
`test_bundle_combiner.py`, `test_champion_guard.py`,
`test_fleet_scheduler.py`, `test_search_role.py`,
`test_ingest_champion.py`, and an E2E plumbing test
(`test_e2e_search.py`) that runs the full baseline → agg_search →
bundle_combine → champion-guard → ingest loop without a GPU, using
mocked `@register_adapter` adapters.

## 3.5.0

Multi-host robustness + observability pass. Ships 8 independently-tested
fixes to keep the autopilot stable in 2+ host FSx-shared deployments.

### Added

- **Cross-backend LLM fallback (`ddddf10`).** `orze-claude` shim now
  honors `ORZE_CLAUDE_FALLBACK=gemini` and falls back to `orze-gemini`
  on subscription-or-API exhaustion. Quota-signal detection extended:
  exit code 42, "quota exceeded", "rate limit", "insufficient
  credits" all trigger the fallback path. Priority order is
  subscription → API → gemini, covered by the new priority-order
  test in `7c2df47`.
- **Flock-based leader election (`34d242f`).** Multi-host deployments
  sharing an FSx state directory now use `fcntl.flock` on
  `results/.orze_leader.lock` to pick exactly one leader per
  role-class cycle. Followers log `follower mode: skipping role <X>`
  and skip leader-only roles (professor, thinker, data_analyst). The
  lock is released on graceful shutdown or process death.
- **`analyst_bridge` agent (`b624e74`).** New agent that converts
  data-analyst insights into queued ideas carrying `origin:analyst`
  metadata. Closes the loop from `_analyst_insights.md` to
  `idea_lake.db` without requiring professor re-ingestion.
- **`orze rebuild-state --all-hosts` (`ebba7e6`, `d7186ea`).**
  Reconstructs `best_idea_id` from `idea_lake.db` and per-idea
  `metrics.json` on startup and on-demand. Covers the case where
  a crashed host left `best_idea_id = null`.

### Fixed

- **Ideas.md corruption detector respects designed wipes (`f57fb1f`).**
  The post-ingest wipe (0 → full regenerate cycle) is no longer
  flagged as corruption. Detector now requires both a shrink AND an
  unexpected timing pattern before restoring from backup.
- **Rolling-window compactor for oversized prompt files (`71c310d`).**
  `_retrospection.txt` and `_skill_composed_research.md` are trimmed
  to a configurable cap (default 150 KB) on every role launch,
  keeping the tail (most recent entries). Prevents 600 KB+ files
  from silently inflating prompt budget.
- **Dedicated stale-DB relocator (`3641a1e`).** Zero-byte `*.db`
  files at the results root (common artifact of a crash during
  sqlite init) are moved to `results/_stale_dbs/` at daemon start,
  instead of being repeatedly re-opened and re-leaked.

### Tested

- Subscription → API → gemini priority order covered end-to-end in
  the shim test suite (`7c2df47`).

### Upgrade notes

Daemons running 3.4.x must be restarted to pick up the fixes — they
cache module code in-process. After restart, run
`orze rebuild-state --results <results> --all-hosts` once to rebuild
`best_idea_id` from the idea lake.

---

## 3.4.11

### Added

- **Role stall detection** (`role_stall_minutes`, default 0 / off in
  core; orze-pro ≥ 0.7.7 opts projects in at 5 min).
  `check_active_roles` now kills a running role whose log file hasn't
  grown for the configured number of minutes, even if the wall-clock
  timeout hasn't elapsed. Mirrors the existing experiment-side
  `stall_minutes` pattern (same `_last_log_size` / `_stall_since`
  semantics). Catches `claude -p` hangs that produce a 0-byte log and
  would otherwise burn the full 20-min role timeout. Emits
  `[ROLE STALL]` (distinct from `[ROLE TIMEOUT]`) so the observability
  signal survives. Reuses `OUTCOME_TIMEOUT` — existing callers don't
  need to branch on a new outcome.

### Fixed

- **`_ideas_were_modified` now credits cross-daemon consumption via
  ideas.md mtime.** `3.4.10`'s `ideas_consumed_during_run` counter
  only works within a single daemon: the consumption phase walks
  `self.active_roles` and can't reach a `RoleProcess` owned by a
  different daemon. In multi-daemon deployments (shared ideas.md on
  FSx, per-daemon `active_roles`), daemon A consuming daemon B's
  research output left B's counter at 0, and the size/count fallback
  also failed because `current_size == pre_size` by coincidence after
  the wipe. Observed ≥9 consecutive soft-failure warnings on a
  productive role in a 2-daemon deployment.

  Fix: snapshot `ideas.md` mtime at role launch into a new
  `RoleProcess.ideas_md_mtime_pre` field. When the consumption-counter
  short-circuit misses, `_ideas_were_modified` checks whether mtime
  advanced past the snapshot. mtime lives on the shared file so every
  daemon sees the same value. Same generosity trade-off as 3.4.10 —
  crediting an idle role is preferable to false-failing a productive
  one.

### Refactored

- **LLM-CLI shim dispatcher** (`orze.shims.llm`). Extracts the
  retry-with-API-key-fallback loop into a single module with a
  `BACKENDS: dict[str, BackendSpec]` registry. Adding a new console
  entry (`orze-codex`, `orze-gemini`, …) is a one-line `BACKENDS`
  addition + one-line `pyproject.toml` entry — no new Python file.
  Behaviour unchanged for the Claude backend; `orze.shims.claude`
  remains as a back-compat re-export.

## 3.4.10

### Fixed

- **`_ideas_were_modified` no longer false-flags research roles that
  append ideas that get consumed mid-run.** Previously, if a research
  agent appended N ideas and orze's consumption phase ingested them
  into `idea_lake.db` (wiping `ideas.md` back to the header) *before*
  the role process exited, the post-exit check compared
  `current_size == pre_size` and `current_count == pre_count` (both
  reflecting the post-consumption state) and returned False → the
  role tripped `OUTCOME_SOFT_FAILURE` and logged
  `"ideas.md was not modified"` despite having produced useful output.
  Over enough cycles this bumped the consecutive-soft-failure counter
  toward the 5/5 circuit breaker.

  Fix: track `ideas_consumed_during_run` per `RoleProcess`. When the
  consumption phase ingests ideas and a research-writer role is still
  running, increment its counter by the ingested count.
  `_ideas_were_modified` short-circuits to True if that counter is
  non-zero, crediting the role for ideas it appended even after the
  file has been wiped.

## 3.4.9

### Fixed

- **`orze stop` output no longer misleads on clean or single-daemon
  hosts.** Previously the command printed "No orchestrator PID file
  found" on the same run where it went on to kill an untracked orze
  process found via `pgrep`, so the first line read like "nothing to
  do." It now prints a single summary:
  - `Nothing to stop — no PID file, no matching process` (clean host)
  - `No PID file but found and killed N untracked orze process(es)…`
  "additional orze process" is renamed to "untracked orze process" —
  it wasn't additional to anything when the PID file was missing.

## 3.4.8

### Fixed

- **`orze/__init__.py:__version__` now reads from `importlib.metadata`
  instead of a hardcoded string.** 3.4.7 shipped with the pyproject
  version bumped to 3.4.7 but `__init__.py` still hardcoding `"3.4.6"`.
  `upgrade.check_pypi` compares `parse_version(latest) > parse_version(__version__)`
  to decide whether to auto-upgrade; with `__version__ = "3.4.6"` and
  PyPI serving `3.4.7`, the check always fired, pip reinstalled no-op,
  and `_restart_in_place` relaunched via `os.execv` — producing an
  infinite restart loop (observed ~8s per cycle on two deployments).
  Reading from installed metadata means the runtime version string
  can never desync from the wheel on disk again.

## 3.4.7

### Fixed

- **`cleanup_orphans` no longer leaves cross-host dead claims stuck
  forever.** The previous logic only reclaimed directories that had
  `claim.json` AND *no* `metrics.json`. A training that wrote partial
  metrics mid-run (e.g. per-epoch snapshots) and then crashed — exactly
  what happens when a second orze host dies mid-training — left
  `metrics.json` on disk, so the directory was skipped by cleanup even
  though the lake still said `status='running'`. Queue slots leaked
  silently.

  Cleanup now has a second branch: when `claim.json` exists AND
  `metrics.json` exists AND the lake reports `status='running'` AND
  last activity is older than `orphan_timeout_hours`, it removes just
  `claim.json` and resets the lake status to `queued`. The directory
  itself (partial metrics, checkpoints, logs) is preserved for
  inspection. Ideas with status `completed`/`failed`/`skipped` remain
  untouched.

  Observed impact: a second node that died 23h earlier had 9 claims
  stuck running; this release reclaims them without losing partial
  work.

## 3.4.6

### Breaking

- **`check_active_roles` now returns `list[tuple[role_name, Outcome]]`**
  instead of `list[tuple[role_name, str]]`. `Outcome` is a small enum
  (`OK`, `SOFT_FAILURE`, `TIMEOUT`, `ERROR`); use `is_success(outcome)`
  when the caller only cares whether the cycle produced useful output.
  Strategy roles that don't append to `ideas.md` (professor, thinker,
  data_analyst, engineer, code_evolution) no longer trip the
  ideas-modified soft-failure check — they set
  `rp.writes_ideas_file = False` at launch and are judged purely on
  exit code.

### Changed

- **`orze.agents.bug_fixer` renamed to `orze.agents.watchdog`.** The
  legacy `bug_fixer` role has been retired from scaffolding (`orze
  --init` no longer materializes `BUG_FIXER_RULES.md`) and from the
  scheduler role registry. Existing projects with `roles.bug_fixer:`
  in `orze.yaml` keep working but receive a one-time warning suggesting
  migration to the watchdog daemon.

## 3.4.5

### Added

- **`metric_harvest.inference_model` / `inference_timeout` / `claude_bin`**
  are now honored by the orchestrator's pattern_inferrer wiring.
  Previously the inferrer was called with hardcoded `model="haiku"`
  / `timeout=60`. Override in `orze.yaml`:
  ```yaml
  metric_harvest:
    inference_model: sonnet        # haiku (default) / sonnet / opus
    inference_timeout: 120         # seconds (default 60)
    claude_bin: /opt/claude        # (default: PATH lookup)
  ```

## 3.4.4

### Fixed

- **Tighter `_log_has_training_signal` gate.** 3.4.3's gate used two
  separate regexes — an eval-keyword check and a numeric check —
  anywhere in the log. That matched dataset-stats lines like
  `"Train: 1500, Test: 1344"` plus a completely unrelated `0.95`
  from a config echo, producing a false "ready for inference"
  signal. New `_EVAL_LINE_PATTERN` requires both signals on the
  same line and at least two decimal digits (filters out integers
  like sample counts).

- **`phases.py`: thread `flat_cfg` into SOP validator.** The dict-
  /list-stripped config (and the repair of LLM-collapsed keys like
  `"epochs: 40": None` → `{"epochs": 40}`) was being built but not
  handed to the SOP `validate_idea` step, which re-fetched the raw
  config and then flagged the collapsed key as unrecognized. Auto-
  patched by the engineer SOP on the live daemon at 16:20 UTC and
  carried into this release.

## 3.4.3

### Fixed

- **`metric_harvester` inference gating and empty-cache TTL.**
  3.4.2's harvester would invoke the LLM pattern_inferrer on any
  idea whose regex missed — including warmup-only logs that hadn't
  yet produced metrics. The inferrer correctly returned `[]` for
  those, but the empty result got cached permanently, blocking
  future (legitimate) inference for that train script forever.

  Two guards added:

  1. `_log_has_training_signal(log_text)` pre-check — the inferrer
     is only called on logs that are >= 50 bytes AND contain an
     eval/epoch marker AND contain a plausibly-metric-like number.
     Warmup-only logs skip inference entirely.

  2. Empty cache entries now expire after 30 min. Non-empty
     entries stick around until the train script mtime changes.
     A log that was too early on first harvest will get retried
     automatically on the next 5-min cycle after it grows.

  Existing poisoned `results/_metric_patterns_cache.json` files
  from 3.4.2 should be deleted once on upgrade; they'll regenerate
  cleanly from the fresh gating rules.

## 3.4.2

### Added

- **LLM pattern-inferrer hook in `metric_harvester`.** When the
  built-in regex patterns and user-configured `metric_harvest.patterns`
  all fail to extract the primary metric from a training script's log,
  the harvester can invoke an optional `pattern_inferrer` callable
  that proposes regex patterns for the specific script + metric.
  Results are cached keyed by `(train_script.name, mtime)` — one call
  per new script per edit, never again.

  orze's harvester stays pure regex by default; the LLM-backed
  inferrer ships in orze-pro (`orze_pro.agents.pattern_inference`)
  and is auto-wired when orze-pro is installed. Disable via
  `metric_harvest.llm_fallback: false`.

  Cached patterns are persisted to `results/_metric_patterns_cache.json`.

### Changed

- `harvest_running_ideas()` gained keyword args `pattern_inferrer` and
  `train_script`. Extraction now strips trailing punctuation
  (`.,;:`) from the captured number, so prose-style patterns like
  `NDCG came in at ([0-9.]+)` safely match values followed by a period.

## 3.4.1

### Added

- **`orze.engine.metric_harvester`**: periodic scan of
  `results/idea-*/train_output.log` every 20 iterations (~5 min).
  Extracts best-so-far for the configured `primary_metric` via
  regex patterns and writes `metrics.json`. Closes the telemetry
  gap for training scripts that log per-epoch metrics to stdout
  but never emit `metrics.json` — previously left the leaderboard
  blind to mid-run progress for the entire 4-12h training window.

  Default patterns cover `map`, `accuracy`, `auc`, `f1`, `loss`.
  Customize via:
  ```yaml
  metric_harvest:
    enabled: true            # default
    patterns:
      - "score\\s*=\\s*([0-9.]+)"
    maximize: true           # false for loss-like metrics
  ```

  Only rewrites files it authored (sentinel
  `_source: "harvested_from_log"`); genuine `metrics.json` from
  training scripts is left untouched.

### Fixed

- `fix(phases): don't crash on slot-race between launch and register`
  (merged from 733b80b).
- `fix(config): reject legacy rules_file key on role configs`
  (8cf5003 from 3.4.0 development).

## 3.4.0

### Breaking

- **`rules_file:` role config key removed.** Every `mode: claude` and
  `mode: research` role must now declare a `skills:` list. The
  loader's legacy `rules_file` fallback is gone. `orze --check` now
  surfaces `roles.<name>: mode 'claude' requires 'skills'` for any
  pre-refactor config.

  Migration: replace `rules_file: FOO.md` with `skills: [./FOO.md]`
  (treat the file as a dynamic SOP) or switch to bundled static
  SOPs with `skills: [@sop:<id>]`. See `orze sop list` for the
  available ids shipped by orze-pro.

- **`orze pro bootstrap-professor` subcommand removed.** Professor
  behavior now ships as bundled static SOPs in orze-pro (0.7.0+); no
  per-project LLM bootstrap is needed. Task-specific tailoring is
  authored as dynamic SOPs under `<project>/skills/`.

### Added

- **`orze sop` subcommand**: `list`, `check`, `status`.
  - `list`: enumerate every registered SOP — skills (static + dynamic),
    methods, validators, portfolios. Shows tier and role.
  - `check`: validate SOP wiring (`requires`, `consumed_by`,
    `overrides`). Exits non-zero on errors.
  - `status`: read execution receipts under `results/_receipts/` and
    show per-SOP last-cycle evidence (`yes`/`NO`).
  Handler is delegated to orze-pro via a guarded import so basic orze
  without orze-pro returns a helpful message.

- **`@sop:<name>` loader prefix** in `orze.skills.loader.compose_skills`.
  Resolves bundled SOPs from orze-pro's `orze_pro/sops/` directory.
  Unknown names log a warning and skip (fail-open).

- **`orze.skills.loader.compose_skills` composition features**: `order`
  frontmatter controls composition order (lower first); `trigger`
  frontmatter gates a skill in/out per cycle via the role's
  `_trigger_context` dict.

- **`orze.skills.triggers.evaluate_trigger`**: gate grammar supporting
  `always`, `periodic_research_cycles(N)`, `on_file(path)`, `on_plateau(N)`.
  Unknown expressions fail open.

- **`orze.skills.receipts`**: execution receipts recorded per role run.
  Captures pre-run mtimes of each declared `produces` path and
  computes which SOPs were evidenced (output files changed).

### Changed

- `orze --init` generates orze.yaml using `skills:` lists for every
  role. No more `rules_file:` entries or copies of bundled prompts
  to the project directory.
- `_check_pro_role` in the Pro Feature Check queries the SOP registry
  via `orze_pro.skills.registry.discover_skills` instead of looking
  for `*_RULES.md` files on disk. Reports `N static + M dynamic`.

### Removed

- Every `rules_file:` example from `RULES.md`, `AGENT.md`,
  `skills/ops.skill.md`, `skills/setup.skill.md`. Docs now show
  `skills:` throughout.
- The `agents.professor_bootstrap` extension registration in
  `orze.extensions` (module is deleted in orze-pro 0.7.0).
