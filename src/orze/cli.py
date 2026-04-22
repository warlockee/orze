#!/usr/bin/env python3
"""Orze CLI — GPU experiment orchestrator.

Calling spec:
    python -m orze.cli                          # all GPUs, continuous
    python -m orze.cli -c orze.yaml --gpus 0,1  # with project config
    python -m orze.cli --once                   # one cycle then exit
    python -m orze.cli --report-only            # regenerate report
    python -m orze.cli --init                   # initialize new project
    python -m orze.cli --admin                  # launch admin panel

This module contains only:
    setup_logging()  — configure log format
    main()           — argparse + dispatch (imports from cli_* modules)

Extracted modules:
    cli_star.py   — star prompt (maybe_star)
    cli_demo.py   — template strings (BASELINE_TRAIN_PY, RESEARCH_RULES_TEMPLATE)
    cli_setup.py  — install/uninstall/upgrade helpers
    cli_pro.py    — pro license management
"""

import argparse
import logging
import os
import subprocess
import sys
from pathlib import Path

from orze import __version__
from orze.cli_pro import pro_activate, pro_status, pro_deactivate
from orze.cli_setup import (
    do_uninstall, stop_running_instance, do_upgrade, do_reinstall,
    do_init, do_check,
)
from orze.cli_star import maybe_star
from orze.core.config import load_project_config
from orze.hardware.gpu import detect_all_gpus

logger = logging.getLogger("orze")


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def setup_logging(verbose: bool = False):
    """Configure logging with timestamps."""
    level = logging.DEBUG if verbose else logging.INFO
    fmt = "%(asctime)s [%(levelname)s] %(message)s"
    datefmt = "%Y-%m-%d %H:%M:%S"
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt)


# ---------------------------------------------------------------------------
# sop subcommand (delegates to orze-pro; SOPs are a pro feature)
# ---------------------------------------------------------------------------

def _run_sop_subcommand(args) -> int:
    try:
        from orze_pro.cli_sop import run_sop_subcommand
    except ImportError:
        print("The 'sop' subcommand requires orze-pro. Install orze-pro "
              "to use SOP registry, wiring validation, and execution "
              "status commands.")
        return 2
    return run_sop_subcommand(args)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    from orze.extensions import _find_pro_key
    if not _find_pro_key():
        maybe_star()

    parser = argparse.ArgumentParser(
        description="orze: GPU experiment orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m orze.cli                             # all GPUs, continuous
  python -m orze.cli -c orze.yaml --gpus 0,1     # with project config
  python -m orze.cli --once                      # one cycle then exit
  python -m orze.cli --report-only               # regenerate report
        """,
    )
    parser.add_argument("-V", "--version", action="version",
                        version=f"orze {__version__}")
    parser.add_argument("-c", "--config-file", type=str, default=None,
                        help="Path to orze.yaml project config")
    parser.add_argument("--gpus", type=str, default=None,
                        help="Comma-separated GPU IDs (default: auto-detect)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Max training time in seconds")
    parser.add_argument("--poll", type=int, default=None,
                        help="Seconds between iterations")
    parser.add_argument("--once", action="store_true",
                        help="Run one cycle and exit")
    parser.add_argument("--stop", action="store_true",
                        help="Gracefully stop a running orze instance")
    parser.add_argument("--restart", action="store_true",
                        help="Stop the running instance and start a new one")
    parser.add_argument("--disable", action="store_true",
                        help="Stop and persistently disable Orze (survives restarts)")
    parser.add_argument("--enable", action="store_true",
                        help="Remove persistent disable flag to allow Orze to run")
    parser.add_argument("--report-only", action="store_true",
                        help="Only regenerate report")
    parser.add_argument("--role-only", type=str, default=None, metavar="NAME",
                        help="Run a single agent role once and exit")
    parser.add_argument("--research-only", action="store_true",
                        help="Alias for --role-only research")
    parser.add_argument("--ideas-md", type=str, default=None,
                        help="Path to ideas markdown file")
    parser.add_argument("--base-config", type=str, default=None,
                        help="Path to base config YAML")
    parser.add_argument("--results-dir", type=str, default=None,
                        help="Directory for results")
    parser.add_argument("--train-script", type=str, default=None,
                        help="Training script to run per idea")
    parser.add_argument("--init", nargs="?", const="__ask__", default=None, metavar="PATH",
                        help="Initialize a new orze project (default: current directory)")
    parser.add_argument("--admin", action="store_true",
                        help="Launch admin panel instead of farm loop")
    parser.add_argument("--upgrade", action="store_true",
                        help="Upgrade orze to the latest version from PyPI")
    parser.add_argument("--reinstall", action="store_true",
                        help="Deep clean + fresh install: uninstall from every "
                             "reachable Python env, purge stale dist-info and "
                             "__pycache__, reinstall, verify single clean version, "
                             "restart. Fixes drift from partial upgrades.")
    parser.add_argument("--reinstall-orze-version", type=str, default=None,
                        metavar="VER", help="Pin orze version for --reinstall")
    parser.add_argument("--reinstall-pro-version", type=str, default=None,
                        metavar="VER", help="Pin orze-pro version for --reinstall")
    parser.add_argument("--reinstall-extra-index-url", type=str, default=None,
                        metavar="URL",
                        help="Extra pip index URL for --reinstall (e.g. private PyPI)")
    parser.add_argument("--no-restart", action="store_true",
                        help="Skip restart after --reinstall")
    parser.add_argument("--check", action="store_true",
                        help="Validate config, check files, API keys, GPUs, .env — then exit")
    parser.add_argument("--uninstall", action="store_true",
                        help="Full uninstall: stop orze, remove runtime files, "
                             "pip uninstall — keeps only research results")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Debug logging")

    # --- subcommands ---
    subparsers = parser.add_subparsers(dest="command")

    # stop
    stop_parser = subparsers.add_parser(
        "stop", help="Stop orze: kill orchestrator + children, "
                     "disable watchdog, clean up GPUs")
    stop_parser.add_argument("-c", "--config-file", type=str, default=None,
                             help="Path to orze.yaml")
    stop_parser.add_argument("--timeout", type=int, default=60,
                             help="Timeout for child processes (default: 60)")

    # start
    start_parser = subparsers.add_parser(
        "start", help="Start orze as a background daemon")
    start_parser.add_argument("-c", "--config-file", type=str, default=None,
                              help="Path to orze.yaml")
    start_parser.add_argument("--gpus", type=str, default=None,
                              help="Comma-separated GPU IDs (default: auto-detect)")
    start_parser.add_argument("--timeout", type=int, default=None,
                              help="Max training time per job in seconds")
    start_parser.add_argument("--foreground", action="store_true",
                              help="Run in foreground instead of daemonizing")

    # restart
    restart_parser = subparsers.add_parser(
        "restart", help="Stop then start orze")
    restart_parser.add_argument("-c", "--config-file", type=str, default=None,
                                help="Path to orze.yaml")
    restart_parser.add_argument("--gpus", type=str, default=None,
                                help="Comma-separated GPU IDs (default: auto-detect)")
    restart_parser.add_argument("--timeout", type=int, default=60,
                                help="Timeout for child processes (default: 60)")
    restart_parser.add_argument("--foreground", action="store_true",
                                help="Run in foreground after restart")

    # service
    svc_parser = subparsers.add_parser("service", help="Manage orze watchdog service")
    svc_sub = svc_parser.add_subparsers(dest="service_action")

    svc_install = svc_sub.add_parser("install", help="Install watchdog service")
    svc_install.add_argument("-c", "--config-file", type=str, default="orze.yaml",
                             help="Path to orze.yaml")
    svc_install.add_argument("--method", choices=["auto", "crontab", "systemd"],
                             default="auto", help="Service method (default: auto)")
    svc_install.add_argument("--stall-threshold", type=int, default=1800,
                             help="Seconds before heartbeat considered stale (default: 1800)")

    svc_sub.add_parser("uninstall", help="Uninstall watchdog service")
    svc_sub.add_parser("status", help="Show watchdog service status")

    svc_logs = svc_sub.add_parser("logs", help="Show watchdog logs")
    svc_logs.add_argument("-n", type=int, default=50,
                          help="Number of log lines (default: 50)")

    # pro
    # reset
    reset_parser = subparsers.add_parser(
        "reset", help="Reset idea lake: purge failed/stale ideas for a fresh start")
    reset_parser.add_argument("-c", "--config-file", type=str, default=None)
    reset_parser.add_argument("--failed", action="store_true",
                              help="Purge all failed ideas")
    reset_parser.add_argument("--all", action="store_true",
                              help="Purge ALL non-completed ideas (queued + failed + partial)")
    reset_parser.add_argument("--full", action="store_true",
                              help="Wipe entire idea lake (backup created)")
    reset_parser.add_argument("-y", "--yes", action="store_true",
                              help="Skip confirmation prompt")

    # result — register external/manual experiment results
    result_parser = subparsers.add_parser(
        "result", help="Register external experiment results so professor/research agents see them")
    result_sub = result_parser.add_subparsers(dest="result_action")
    result_add = result_sub.add_parser("add", help="Add a manual result")
    result_add.add_argument("--name", required=True, help="Experiment name (e.g. riskprop_repro_ep10)")
    result_add.add_argument("--map", type=float, required=True, help="mAP score")
    result_add.add_argument("--epoch", type=int, default=None, help="Best epoch")
    result_add.add_argument("--pipeline", type=str, default="manual", help="Pipeline name")
    result_add.add_argument("--notes", type=str, default="", help="Notes about the result")
    result_add.add_argument("--source-dir", type=str, default=None,
                            help="Source code directory for method analysis (writes _methods/<name>.yaml)")
    result_add.add_argument("-c", "--config-file", type=str, default=None)
    result_sub.add_parser("list", help="List all manual results")
    result_rm = result_sub.add_parser("rm", help="Remove a manual result by name")
    result_rm.add_argument("name", help="Experiment name to remove")
    result_rm.add_argument("-c", "--config-file", type=str, default=None)

    pro_parser = subparsers.add_parser("pro", help="Manage orze-pro license")
    pro_sub = pro_parser.add_subparsers(dest="pro_action")
    pro_activate_parser = pro_sub.add_parser("activate", help="Activate orze-pro with a license key")
    pro_activate_parser.add_argument("key", nargs="?", default=None, help="License key (or enter interactively)")
    pro_sub.add_parser("status", help="Show orze-pro license status")
    pro_deactivate_parser = pro_sub.add_parser("deactivate", help="Remove saved license key")
    pro_deactivate_parser.add_argument("-y", "--yes", action="store_true", help="Skip confirmation")

    # --- sop: inspect and validate SOP skills ---
    sop_parser = subparsers.add_parser(
        "sop", help="Inspect SOP skills")
    sop_sub = sop_parser.add_subparsers(dest="sop_command")
    sop_list_p = sop_sub.add_parser(
        "list", help="List all SOPs (skills + validators + methods + portfolios)")
    sop_list_p.add_argument("--project-root", default=".",
                            help="Project root (default: cwd)")
    sop_list_p.add_argument("--results-dir", default="orze_results",
                            help="Results dir for Tier 2 YAML SOPs "
                                 "(default: orze_results)")
    sop_check_p = sop_sub.add_parser(
        "check", help="Validate SOP wiring (requires/consumed_by/overrides)")
    sop_check_p.add_argument("--project-root", default=".")
    sop_status_p = sop_sub.add_parser(
        "status",
        help="Show last-run execution evidence per SOP from receipts")
    sop_status_p.add_argument("--project-root", default=".")
    sop_status_p.add_argument("--results-dir", default="orze_results")

    # --- rebuild-state: rebuild best_idea_id from idea_lake.db ---
    rebuild_parser = subparsers.add_parser(
        "rebuild-state",
        help="Rebuild best_idea_id + completions_since_best from idea_lake.db",
    )
    rebuild_parser.add_argument("-c", "--config-file", type=str, default=None)
    rebuild_parser.add_argument("--results", type=str, default=None,
                                help="Results dir (default: from config)")
    rebuild_parser.add_argument("--overwrite", action="store_true",
                                help="Overwrite even if best_idea_id is set")
    rebuild_parser.add_argument("--all-hosts", action="store_true",
                                help="Update .orze_state_*.json for every "
                                     "host (shared FSx multi-daemon case)")

    # --- catalog: artifact catalog (F9) ---
    catalog_parser = subparsers.add_parser(
        "catalog",
        help="Manage the ArtifactCatalog (ckpts / preds NPZs index)",
    )
    catalog_sub = catalog_parser.add_subparsers(dest="catalog_action")
    catalog_scan = catalog_sub.add_parser("scan", help="Scan a results dir")
    catalog_scan.add_argument("--results-dir", required=True,
                              help="Root directory to walk for artifacts")
    catalog_scan.add_argument("--db", default=None,
                              help="Artifact DB path "
                                   "(default: <results>/idea_lake_artifacts.db)")
    catalog_scan.add_argument("--no-hash", action="store_true",
                              help="Skip ckpt hashing (faster, but ckpt_sha "
                                   "won't be available for bundling)")
    catalog_scan.add_argument("--limit", type=int, default=None,
                              help="Stop after N files (debug)")
    catalog_ls = catalog_sub.add_parser("list", help="List artifacts")
    catalog_ls.add_argument("--db", required=True)
    catalog_ls.add_argument("--kind", default=None)
    catalog_ls.add_argument("--ckpt-sha", default=None)

    # --- ingest-champion (F16): retroactive champion record ---
    ingest_parser = subparsers.add_parser(
        "ingest-champion",
        help="Retroactively ingest a manual champion bundle into idea_lake"
    )
    ingest_parser.add_argument("--results-dir", required=True)
    ingest_parser.add_argument("--idea-id", default="idea-champion-0905")
    ingest_parser.add_argument("--config", default=None,
        help="Path to _champion_config.json "
             "(defaults to <results_dir>/_champion_config.json)")
    ingest_parser.add_argument("--project-root", default=None)

    # --- rebuild-lake (v4.0): rebuild idea_lake.db from results dirs ---
    rl_parser = subparsers.add_parser(
        "rebuild-lake",
        help="Rebuild idea_lake.db from existing results directories")
    rl_parser.add_argument("--results-dir", default="orze_results")
    rl_parser.add_argument("--db", default="idea_lake.db")

    # --- manual-notify (v4.0): send a manual report ---
    mn_parser = subparsers.add_parser(
        "manual-notify",
        help="Send a manual status notification report")
    mn_parser.add_argument("-c", "--config", default="orze.yaml")

    # --- hf-discover (v4.0): query HuggingFace Hub for models ---
    hf_parser = subparsers.add_parser(
        "hf-discover",
        help="Query the HuggingFace Hub for models matching criteria")
    hf_parser.add_argument("--pipeline-tag", default="image-feature-extraction")
    hf_parser.add_argument("--min-downloads", type=int, default=50000)
    hf_parser.add_argument("--limit", type=int, default=20)

    args = parser.parse_args()

    setup_logging(args.verbose)

    # --- subcommand dispatch ---
    command = getattr(args, "command", None)

    if command == "sop":
        return _run_sop_subcommand(args)

    if command == "catalog":
        from orze.artifact_catalog import ArtifactCatalog, cli_scan
        action = getattr(args, "catalog_action", None)
        if action == "scan":
            return cli_scan(args)
        if action == "list":
            cat = ArtifactCatalog(args.db)
            rows = (cat.by_ckpt_sha(args.ckpt_sha) if args.ckpt_sha
                    else cat.list_by_kind(args.kind) if args.kind
                    else [cat.get(r[0]) for r in cat.conn.execute(
                        "SELECT path FROM artifacts ORDER BY created_at DESC")])
            for r in rows:
                if not r:
                    continue
                print(f"[{r['kind']:10s}] sha={r.get('ckpt_sha') or '-':>16s} "
                      f"val={r.get('metric_val')} {r['path']}")
            cat.close()
            return 0
        print("usage: orze catalog {scan,list} …")
        return 2

    if command == "ingest-champion":
        from orze.agents.ingest_champion import ingest
        info = ingest(
            Path(args.results_dir),
            idea_id=args.idea_id,
            config_path=Path(args.config) if args.config else None,
            project_root=Path(args.project_root) if args.project_root else None,
        )
        import json as _json
        print(_json.dumps(info, indent=2))
        return 0

    if command == "rebuild-lake":
        from orze.rebuild_lake import rebuild
        rebuild(Path(args.results_dir), Path(args.db))
        return 0

    if command == "manual-notify":
        from orze.manual_notify import main as _mn_main
        import sys as _sys
        _sys.argv = ["orze manual-notify", "-c", args.config]
        _mn_main()
        return 0

    if command == "hf-discover":
        from orze.hf_discover import search_models
        import json as _json
        models = search_models(pipeline_tag=args.pipeline_tag,
                               min_downloads=args.min_downloads,
                               limit=args.limit)
        print(_json.dumps(models, indent=2))
        return 0

    if command == "rebuild-state":
        from orze.engine.rebuild_state import rebuild_state_file
        cfg = load_project_config(args.config_file)
        results_dir = Path(args.results or cfg.get("results_dir", "orze_results"))
        summary = rebuild_state_file(results_dir, cfg,
                                     overwrite=args.overwrite,
                                     all_hosts=args.all_hosts)
        print(f"primary_metric: {summary['primary_metric']}")
        print(f"best_idea_id: {summary['best_idea_id']}")
        print(f"completions_since_best: {summary['completions_since_best']}")
        print(f"previous_best_idea_id: {summary['previous_best_idea_id']}")
        if summary['wrote_state_file']:
            print(f"Wrote: {summary['state_file']}")
            if summary.get("updated_hosts"):
                print(f"Updated hosts: {', '.join(summary['updated_hosts'])}")
        else:
            print("(state file already had best_idea_id; "
                  "pass --overwrite to force)")
        return

    if command == "stop":
        from orze.lifecycle import do_stop
        cfg = load_project_config(args.config_file)
        do_stop(cfg, timeout=args.timeout)
        return

    if command == "start":
        from orze.lifecycle import do_start
        cfg = load_project_config(args.config_file)
        if args.timeout is not None:
            cfg["timeout"] = args.timeout
        config_path = args.config_file or cfg.get("_config_path", "orze.yaml")
        do_start(cfg, foreground=args.foreground, config_path=config_path,
                 gpus=args.gpus, timeout=args.timeout)
        return

    if command == "restart":
        from orze.lifecycle import do_restart
        cfg = load_project_config(args.config_file)
        config_path = args.config_file or cfg.get("_config_path", "orze.yaml")
        do_restart(cfg, timeout=args.timeout, foreground=args.foreground,
                   config_path=config_path, gpus=args.gpus)
        return

    if command == "reset":
        import sqlite3, shutil
        cfg = load_project_config(args.config_file)
        db_path = Path(cfg.get("idea_lake_db") or Path(cfg.get("results_dir", "orze_results")) / "idea_lake.db")
        if not db_path.exists():
            print("No idea_lake.db found.")
            return

        conn = sqlite3.connect(str(db_path))
        c = conn.cursor()

        if args.full:
            if not args.yes:
                c.execute("SELECT COUNT(*) FROM ideas")
                total = c.fetchone()[0]
                resp = input(f"Wipe entire idea lake ({total} ideas)? Backup will be created. [y/N] ")
                if resp.lower() != "y":
                    print("Aborted.")
                    return
            backup = db_path.with_suffix(".db.bak")
            shutil.copy2(db_path, backup)
            c.execute("DELETE FROM ideas")
            print(f"Wiped {c.rowcount} ideas. Backup: {backup}")
        elif args.all:
            c.execute("DELETE FROM ideas WHERE status IN ('queued', 'failed', 'partial', 'running')")
            print(f"Purged {c.rowcount} non-completed ideas.")
        elif args.failed:
            c.execute("DELETE FROM ideas WHERE status = 'failed'")
            print(f"Purged {c.rowcount} failed ideas.")
        else:
            # Default: show status summary
            c.execute("SELECT status, COUNT(*) FROM ideas GROUP BY status")
            for row in c.fetchall():
                print(f"  {row[0]}: {row[1]}")
            print("\nUse --failed, --all, or --full to purge.")

        conn.commit()
        conn.close()

        # Also clear pause sentinel — stale failures shouldn't block research
        results_dir = Path(cfg.get("results_dir", "orze_results"))
        pause_file = results_dir / ".pause_research"
        if pause_file.exists():
            pause_file.unlink()
            print("Cleared .pause_research sentinel.")

        return

    if command == "result":
        import json as _json
        action = getattr(args, "result_action", None)
        cfg = load_project_config(getattr(args, "config_file", None))
        results_dir = Path(cfg.get("results_dir", "orze_results"))
        manual_path = results_dir / "_manual_results.json"

        if action == "add":
            entries = []
            if manual_path.exists():
                try:
                    entries = _json.loads(manual_path.read_text(encoding="utf-8"))
                except (ValueError, OSError):
                    pass
            # Remove existing entry with same name (update)
            entries = [e for e in entries if e.get("name") != args.name]
            entry = {"name": args.name, "map": args.map, "source": "manual"}
            if args.epoch is not None:
                entry["epoch"] = args.epoch
            if args.pipeline != "manual":
                entry["pipeline"] = args.pipeline
            if args.notes:
                entry["notes"] = args.notes
            entries.append(entry)
            entries.sort(key=lambda e: float(e.get("map", 0) or 0), reverse=True)
            manual_path.write_text(_json.dumps(entries, indent=2) + "\n",
                                   encoding="utf-8")
            print(f"Registered: {args.name} (mAP={args.map})")
            print(f"  Saved to {manual_path}")
            # SOP: extract method spec from source code (orze-pro)
            if getattr(args, "source_dir", None):
                from orze.extensions import get_extension
                _sops = get_extension("sops")
                if _sops:
                    method_path = _sops.analyze_method(args.name, Path(args.source_dir),
                                                        results_dir)
                else:
                    method_path = None
                    print("  (Install orze-pro for method analysis)")
                if method_path:
                    print(f"  Method spec written to {method_path}")
            # SOP: trigger professor to analyze the new result and create portfolio
            trigger_path = results_dir / "_trigger_professor"
            trigger_path.write_text(
                f"new_external_result: {args.name} (mAP={args.map}). "
                f"Read the method spec at results/_methods/{args.name}.yaml, "
                f"enrich it with exact loss formulas from the source code, "
                f"then write a portfolio to results/_portfolios/ that ports "
                f"this method to all viable backbones.",
                encoding="utf-8")
            print(f"  Professor triggered to analyze and create portfolio.")
        elif action == "rm":
            if manual_path.exists():
                entries = _json.loads(manual_path.read_text(encoding="utf-8"))
                before = len(entries)
                entries = [e for e in entries if e.get("name") != args.name]
                if len(entries) < before:
                    manual_path.write_text(_json.dumps(entries, indent=2) + "\n",
                                           encoding="utf-8")
                    print(f"Removed: {args.name}")
                else:
                    print(f"Not found: {args.name}")
            else:
                print("No manual results registered.")
        else:
            # list
            if manual_path.exists():
                entries = _json.loads(manual_path.read_text(encoding="utf-8"))
                if entries:
                    print(f"{'Name':<35} {'mAP':>8}  {'Notes'}")
                    print("-" * 80)
                    for e in entries:
                        print(f"{e.get('name','?'):<35} {e.get('map','?'):>8}  {e.get('notes','')[:40]}")
                else:
                    print("No manual results.")
            else:
                print("No manual results registered yet.")
                print(f"  Use: orze result add --name <name> --map <score>")
        return

    if command == "pro":
        action = getattr(args, "pro_action", None)
        if action == "activate":
            pro_activate(getattr(args, "key", None))
        elif action == "status":
            pro_status()
        elif action == "deactivate":
            pro_deactivate(force=getattr(args, "yes", False))
        else:
            parser.parse_args(["pro", "--help"])
        return

    if command == "service":
        action = getattr(args, "service_action", None)
        if action == "install":
            from orze.service.install import install
            install(args.config_file, method=args.method,
                    stall_threshold=args.stall_threshold)
        elif action == "uninstall":
            from orze.service.install import uninstall
            uninstall()
        elif action == "status":
            from orze.service.status import show_status
            show_status()
        elif action == "logs":
            from orze.service.status import show_logs
            show_logs(n=args.n)
        else:
            parser.parse_args(["service", "--help"])
        return

    # Load project config, then apply CLI overrides
    cfg = load_project_config(args.config_file)
    cfg["_config_path"] = args.config_file or "orze.yaml"  # stored for mode: research

    # --admin: launch web panel
    if args.admin:
        from orze.admin.server import run_admin
        run_admin(cfg)
        return

    # --upgrade: upgrade orze from PyPI (stops + restarts if running)
    if args.upgrade:
        do_upgrade(cfg)
        return

    # --reinstall: deep-clean reinstall (fixes partial-upgrade drift)
    if args.reinstall:
        do_reinstall(
            cfg,
            orze_version=args.reinstall_orze_version,
            pro_version=args.reinstall_pro_version,
            extra_index_url=args.reinstall_extra_index_url,
            no_restart=args.no_restart,
        )
        return

    # --uninstall: full cleanup, keep only research results
    if args.uninstall:
        do_uninstall(cfg)
        return

    # --init: initialize a new project
    if args.init is not None:
        do_init(args.init)
        return

    # --check: validate config and environment, then exit
    if args.check:
        do_check(cfg)
        return

    # Apply CLI overrides
    if args.timeout is not None:
        cfg["timeout"] = args.timeout
    if args.poll is not None:
        cfg["poll"] = args.poll
    if args.ideas_md:
        cfg["ideas_file"] = args.ideas_md
    if args.base_config:
        cfg["base_config"] = args.base_config
    if args.results_dir:
        cfg["results_dir"] = args.results_dir
    if args.train_script:
        cfg["train_script"] = args.train_script

    # --stop
    if args.stop:
        import datetime
        from orze.core.fs import atomic_write
        stop_path = Path(cfg["results_dir"]) / ".orze_stop_all"
        atomic_write(stop_path,
                     f"kill {datetime.datetime.now().isoformat()}")
        print(f"Stop signal written to {stop_path}. "
              f"All nodes sharing this results directory will stop within "
              f"~30 seconds. Training, evaluation, and research processes "
              f"will be terminated (SIGTERM, then SIGKILL after 10s). "
              f"The sentinel is cleared automatically on next startup.")
        return

    # --restart: stop running instance, then continue to start a new one
    if args.restart:
        stop_running_instance(Path(cfg["results_dir"]))
        print("Starting new orze instance...")

    # --disable
    if args.disable:
        import datetime
        from orze.core.fs import atomic_write
        disable_path = Path(cfg["results_dir"]) / ".orze_disabled"
        atomic_write(disable_path, f"Disabled at {datetime.datetime.now().isoformat()}")
        print(f"Orze disabled. Remove {disable_path} to re-enable.")
        return

    # --enable
    if args.enable:
        disable_path = Path(cfg["results_dir"]) / ".orze_disabled"
        if disable_path.exists():
            disable_path.unlink()
            print("Orze re-enabled.")
        else:
            print("Orze was not disabled.")
        return

    # --report-only
    if args.report_only:
        from orze.core.ideas import parse_ideas
        from orze.reporting.leaderboard import update_report
        ideas = parse_ideas(cfg["ideas_file"])
        results_dir = Path(cfg["results_dir"])
        lake = None
        lake_path = Path(cfg["idea_lake_db"])
        if lake_path.exists():
            from orze.idea_lake import IdeaLake
            lake = IdeaLake(str(lake_path))
        update_report(results_dir, ideas, cfg, lake=lake)
        print("Report updated.")
        return

    # --research-only is an alias for --role-only research
    if args.research_only:
        args.role_only = "research"

    # Detect GPUs
    if args.gpus:
        gpu_ids = [int(g.strip()) for g in args.gpus.split(",")]
    else:
        gpu_ids = detect_all_gpus()

    if not gpu_ids:
        logger.error("No GPUs detected. Use --gpus to specify manually.")
        sys.exit(1)

    # Start admin panel in background thread (unless --role-only or --admin-off)
    if not args.role_only and not getattr(args, 'no_admin', False):
        try:
            import threading
            from orze.admin.server import run_admin as _run_admin_server
            admin_port = int(cfg.get("admin_port") or os.environ.get("ORZE_ADMIN_PORT", "8787"))

            def _admin_thread():
                try:
                    _run_admin_server(cfg, port=admin_port)
                except OSError as e:
                    if "address already in use" in str(e).lower():
                        logger.warning(
                            "Admin port %d already in use by another process. "
                            "Skipping admin panel to avoid killing another instance. "
                            "Set admin_port in orze.yaml to use a different port.",
                            admin_port)
                    else:
                        logger.warning("Admin panel failed to start: %s", e)
                except Exception as e:
                    logger.warning("Admin panel failed to start: %s", e)

            t = threading.Thread(target=_admin_thread, daemon=True)
            t.start()
            logger.info("Admin panel starting on http://0.0.0.0:%d", admin_port)
        except Exception as e:
            logger.warning("Could not start admin panel: %s", e)

    # Launch orchestrator
    from orze.engine.orchestrator import Orze
    orze = Orze(gpu_ids, cfg, once=args.once)

    if args.role_only:
        orze._run_role_once(args.role_only)
    else:
        orze.run()


if __name__ == "__main__":
    main()
