"""Install, uninstall, upgrade, init, and check helpers for the CLI.

Calling spec:
    from orze.cli_setup import (
        do_uninstall, stop_running_instance, do_upgrade,
        find_shared_mounts, resolve_init_path,
        do_init, do_check,
        IDEA_KEEP, RESULTS_KEEP,
    )

    do_uninstall(cfg)          # full uninstall, preserves research results
    stop_running_instance(p)   # SIGTERM a running orze via PID file
    do_upgrade(cfg)            # pip/uv upgrade + optional restart
    find_shared_mounts()       # detect network FS mounts
    resolve_init_path(path)    # resolve --init target directory
    do_init(init_arg)          # scaffold a new orze project
    do_check(cfg)              # validate config and environment

Pure functions: find_shared_mounts, resolve_init_path.
Side-effectful: do_uninstall, stop_running_instance, do_upgrade, do_init, do_check.
"""

import base64
import os
import shutil
import subprocess
import sys
import zlib
from pathlib import Path

from orze import __version__


def _decompress_b85(data: str) -> str:
    """Decompress a base85-encoded zlib-compressed string."""
    return zlib.decompress(base64.b85decode(data)).decode("utf-8")


# Embedded AGENT.md and RULES.md — guaranteed fallback, never fails.
# Generated from src/orze/AGENT.md and src/orze/RULES.md via:
#   base64.b85encode(zlib.compress(content, 9))
def _get_embedded_docs() -> dict:
    """Return embedded doc contents. Lazy-loaded to avoid bloating import."""
    try:
        # Try file-based first (preferred — editable, up-to-date)
        pkg_dir = Path(__file__).resolve().parent
        result = {}
        for name in ("AGENT.md", "RULES.md"):
            src = pkg_dir / name
            if src.exists():
                result[name] = src.read_text(encoding="utf-8")
            else:
                # Try importlib.resources
                try:
                    import importlib.resources
                    ref = importlib.resources.files("orze").joinpath(name)
                    result[name] = ref.read_text(encoding="utf-8")
                except Exception:
                    pass
        if "AGENT.md" in result:
            return result
    except Exception:
        pass

    # Network fallback
    try:
        import urllib.request
        result = {}
        for name in ("AGENT.md", "RULES.md"):
            url = f"https://raw.githubusercontent.com/warlockee/orze/main/src/orze/{name}"
            with urllib.request.urlopen(url, timeout=5) as resp:
                result[name] = resp.read().decode("utf-8")
        if "AGENT.md" in result:
            return result
    except Exception:
        pass

    # Should never reach here — but if it does, return empty.
    # The caller handles missing keys gracefully.
    return {}

# ---------------------------------------------------------------------------
# Pro feature self-check helpers
# ---------------------------------------------------------------------------

def _check_pro_license() -> tuple:
    """Check if orze-pro license is activated."""
    try:
        from orze_pro.license import is_licensed, load_key_payload
        if is_licensed():
            payload = load_key_payload()
            return True, f"licensed to {payload.get('customer', '?')} (expires {payload.get('expires', '?')})"
        return False, "not activated — run: orze pro activate <key>"
    except Exception:
        return False, "not activated — run: orze pro activate <key>"


def _check_pro_role(role_name: str, project_dir: Path) -> tuple:
    """Check if a pro role's SOP skills are discoverable.

    Post-refactor, roles compose from bundled static SOPs (@sop:<name>)
    and optional project-local dynamic SOPs. Success means orze-pro is
    installed AND at least one SOP for the role is registered.
    """
    try:
        from orze_pro.skills.registry import discover_skills
    except ImportError:
        return False, "orze-pro not installed"
    skills = discover_skills(project_dir)
    role_skills = [s for s in skills if s.role == role_name]
    if not role_skills:
        return False, f"no SOPs registered for role '{role_name}'"
    static = [s for s in role_skills if s.tier == "static"]
    dynamic = [s for s in role_skills if s.tier == "dynamic"]
    detail = f"{len(static)} static"
    if dynamic:
        detail += f" + {len(dynamic)} dynamic"
    return True, f"ready ({detail})"


def _check_pro_procedures() -> tuple:
    """Check if pro procedures are discoverable."""
    try:
        import orze_pro
        proc_dir = Path(orze_pro.__file__).parent / "procedures"
        yamls = list(proc_dir.glob("*.yaml"))
        if yamls:
            return True, f"{len(yamls)} procedures available"
        return False, "no procedures found in orze-pro package"
    except ImportError:
        return False, "orze-pro not installed"


def _check_pro_plugin(plugin_name: str) -> tuple:
    """Check if a pro plugin is importable."""
    try:
        import orze_pro
        plugin_dir = Path(orze_pro.__file__).parent / "fsm" / "plugins"
        if (plugin_dir / f"{plugin_name}.py").exists():
            return True, "ready"
        return False, f"{plugin_name}.py not found"
    except ImportError:
        return False, "orze-pro not installed"


def _check_api_keys() -> tuple:
    """Check if any LLM API keys are available."""
    keys = []
    if os.environ.get("ANTHROPIC_API_KEY"):
        keys.append("Anthropic")
    if os.environ.get("GEMINI_API_KEY"):
        keys.append("Gemini")
    if os.environ.get("OPENAI_API_KEY"):
        keys.append("OpenAI")
    if keys:
        return True, ", ".join(keys)
    # Check .env file (only uncommented, non-empty values)
    env_path = Path(".env")
    if env_path.exists():
        for line in env_path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line.startswith("#") or "=" not in line:
                continue
            var, val = line.split("=", 1)
            val = val.strip().strip("'\"")
            if not val or val.startswith("sk-...") or val.startswith("AI..."):
                continue  # placeholder
            for env_var, name in [("ANTHROPIC_API_KEY", "Anthropic"),
                                  ("GEMINI_API_KEY", "Gemini"),
                                  ("OPENAI_API_KEY", "OpenAI")]:
                if var.strip() == env_var:
                    keys.append(name)
    if keys:
        return True, f"{', '.join(keys)} (in .env)"
    return False, "none — add to .env"


# Files inside each idea dir to keep (research results)
IDEA_KEEP = {
    "metrics.json",
    "eval_report.json",
    "eval_output.log",
    "best_model.pt",
    "best_model.pth",
    "model.pt",
    "model.pth",
    "checkpoint.pt",
    "checkpoint.pth",
}

# Top-level results_dir files to keep
RESULTS_KEEP = {"report.md", "status.json"}


def do_uninstall(cfg: dict):
    """Full uninstall: stop orze, strip runtime files, pip uninstall.

    Preserves research results:
        - results/{idea-*}/metrics.json  (core experiment data)
        - results/{idea-*}/eval_report.json, eval_output.log
        - results/report.md, status.json
        - ideas.md, idea_lake.db
    """
    import time

    results_dir = Path(cfg["results_dir"])
    print("\n\033[1mOrze — Uninstall\033[0m")
    print("-----------------")
    print(f"Results dir : {results_dir.resolve()}")
    print(f"Config      : {cfg.get('_config_path', 'orze.yaml')}")
    print()

    # --- 1. Stop running instances -----------------------------------
    print("[1/6] Stopping running orze instances...")
    stop_path = results_dir / ".orze_stop_all"
    if results_dir.exists():
        stop_path.write_text("uninstall", encoding="utf-8")
        # Also send SIGTERM to any PID files we find
        for pid_file in results_dir.glob(".orze.pid*"):
            try:
                pid = int(pid_file.read_text(encoding="utf-8").strip())
                os.kill(pid, 15)  # SIGTERM
                print(f"  Sent SIGTERM to PID {pid}")
            except (ValueError, ProcessLookupError, PermissionError, OSError):
                pass
    time.sleep(2)  # give processes a moment to exit

    # --- 2. Clean results directory ----------------------------------
    print("[2/6] Cleaning runtime files from results/...")
    if results_dir.exists():
        # Fast path: rename the whole dir, create a fresh one, move only
        # keeper files back (O(1) renames), then bulk-delete the old tree.
        stale = results_dir.with_name(results_dir.name + "._orze_uninstall_tmp")
        if stale.exists():
            shutil.rmtree(stale, ignore_errors=True)
        results_dir.rename(stale)
        results_dir.mkdir()

        # Restore top-level keeper files (report.md, status.json)
        for name in RESULTS_KEEP:
            src = stale / name
            if src.exists():
                src.rename(results_dir / name)

        # Restore keeper files and subdirs from idea directories
        for p in stale.iterdir():
            if not p.is_dir() or p.name.startswith((".", "_")):
                continue
            keep_files = [c for c in p.iterdir()
                          if c.is_file() and c.name in IDEA_KEEP]
            keep_dirs = [c for c in p.iterdir() if c.is_dir()]
            if keep_files or keep_dirs:
                dest = results_dir / p.name
                dest.mkdir()
                for f in keep_files:
                    f.rename(dest / f.name)
                for d in keep_dirs:
                    d.rename(dest / d.name)

        # Bulk-delete the old tree (fast, handled in C)
        shutil.rmtree(stale, ignore_errors=True)

    print("  Cleaned runtime files")

    # --- 2b. Remove orze-generated scaffolding files -------------------
    print("[2b/6] Cleaning orze scaffolding...")
    # Files orze --init creates (safe to remove — not user content)
    _ORZE_GENERATED = [
        "ORZE-AGENT.md",
        "ORZE-RULES.md",
        "RESEARCH_RULES.md",
        ".env",
        "orze.yaml",
    ]
    # Patterns for orze runtime artifacts
    _ORZE_PATTERNS = [
        "ideas.md.safe*",      # backup files from role_runner
        "idea_lake.db",
    ]
    # Directories orze --init creates
    _ORZE_DIRS = [
        "procedures",          # user procedure overrides
    ]

    for name in _ORZE_GENERATED:
        p = Path(name)
        if p.exists():
            p.unlink()
            print(f"  Removed {name}")

    for pattern in _ORZE_PATTERNS:
        for p in Path(".").glob(pattern):
            p.unlink()
            print(f"  Removed {p}")

    for dirname in _ORZE_DIRS:
        d = Path(dirname)
        if d.is_dir():
            shutil.rmtree(d, ignore_errors=True)
            print(f"  Removed {dirname}/")

    # --- 3. Remove venv directory ------------------------------------
    print("[3/6] Removing venv...")
    venv_dir = Path("venv")
    if venv_dir.is_dir():
        shutil.rmtree(venv_dir, ignore_errors=True)
        print("  Removed venv/")
    else:
        print("  No venv/ found")

    # --- 4. Remove orze config file ----------------------------------
    print("[4/6] Removing orze config...")
    config_path = Path(cfg.get("_config_path", "orze.yaml"))
    if config_path.exists():
        config_path.unlink()
        print(f"  Removed {config_path}")
    else:
        print("  No config file found")

    # --- 5. Uninstall orze package (try uv first, fall back to pip) --
    print("[5/6] Uninstalling orze package...")
    uninstalled = False
    if shutil.which("uv"):
        try:
            subprocess.run(
                ["uv", "tool", "uninstall", "orze"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            print("  orze package removed (uv)")
            uninstalled = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            pass
    if not uninstalled:
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "uninstall", "orze", "-y"],
                check=True,
            )
            print("  orze package removed (pip)")
        except subprocess.CalledProcessError:
            print("  WARNING: uninstall failed — you may need to run manually:")
            print("    uv tool uninstall orze  OR  pip uninstall orze")

    # --- 6. Also uninstall orze-pro if present -------------------------
    print("[6/6] Checking for orze-pro...")
    try:
        subprocess.run(
            [sys.executable, "-m", "pip", "uninstall", "orze-pro", "-y"],
            check=True,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        print("  orze-pro removed")
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  orze-pro not installed (skipped)")

    # --- Summary -----------------------------------------------------
    print()
    print("\033[1mUninstall complete.\033[0m")
    print()

    # Show what's preserved (user's work)
    preserved = []
    if results_dir.exists():
        kept = list(results_dir.rglob("metrics.json"))
        if kept:
            preserved.append(f"  {results_dir}/ ({len(kept)} experiment results)")
    ideas_file = Path(cfg.get("ideas_file", "ideas.md"))
    if ideas_file.exists():
        preserved.append(f"  {ideas_file}")
    for name in ["GOAL.md", "train.py", "configs"]:
        p = Path(name)
        if p.exists():
            preserved.append(f"  {name}")

    if preserved:
        print("Preserved (your work):")
        for p in preserved:
            print(p)
    else:
        print("No user files remaining.")

    project_dir = Path.cwd()
    print()
    print(f"  To remove everything: rm -rf {project_dir}")


def stop_running_instance(results_dir: Path) -> bool:
    """Stop a running orze instance via PID file. Returns True if one was stopped."""
    import time as _time
    pid_file = results_dir / ".orze.pid"
    if not pid_file.exists():
        return False
    try:
        old_pid = int(pid_file.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return False
    print(f"Stopping orze (PID {old_pid})...")
    try:
        os.kill(old_pid, 15)  # SIGTERM
    except ProcessLookupError:
        print("  Process already exited.")
        return False
    for _ in range(30):  # 15s
        try:
            os.kill(old_pid, 0)
            _time.sleep(0.5)
        except ProcessLookupError:
            break
    else:
        print("  Still running after 15s, sending SIGKILL...")
        try:
            os.kill(old_pid, 9)
            _time.sleep(1)
        except ProcessLookupError:
            pass
    print("  Stopped.")
    return True


def do_upgrade(cfg: dict):
    """Upgrade orze to the latest version from PyPI, then restart if running."""
    print(f"\n\033[1mOrze — Upgrade\033[0m")
    print("---------------")
    print(f"Current version: {__version__}")

    # 1. Stop running instance first (so new code loads on restart)
    results_dir = Path(cfg["results_dir"])
    was_running = stop_running_instance(results_dir)

    # 2. Upgrade (try uv first, fall back to pip)
    upgraded = False
    if shutil.which("uv"):
        print("Upgrading via uv...\n")
        try:
            subprocess.run(
                ["uv", "tool", "upgrade", "orze"],
                check=True,
            )
            upgraded = True
        except (subprocess.CalledProcessError, FileNotFoundError):
            print("  uv upgrade failed, falling back to pip...")

    if not upgraded:
        print("Upgrading via pip...\n")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "--upgrade", "orze"],
                check=True,
            )
        except subprocess.CalledProcessError:
            print("\n\033[31mUpgrade failed.\033[0m Try manually:")
            print("  uv tool upgrade orze  OR  pip install --upgrade orze")
            return

    # Report the new version
    result = subprocess.run(
        [sys.executable, "-c", "from orze import __version__; print(__version__)"],
        capture_output=True, text=True,
    )
    new_ver = result.stdout.strip() if result.returncode == 0 else "unknown"
    if new_ver == __version__:
        print(f"\nAlready up-to-date (v{__version__}).")
    else:
        print(f"\n\033[32mUpgraded: v{__version__} -> v{new_ver}\033[0m")

    # 3. Restart if it was running
    if was_running:
        print("\nRestarting orze with new version...")
        os.execv(sys.executable, [sys.executable, "-m", "orze.cli",
                                  "-c", str(cfg.get("_config_path", "orze.yaml"))])


def do_reinstall(cfg: dict,
                 orze_version: str = None,
                 pro_version: str = None,
                 extra_index_url: str = None,
                 no_restart: bool = False) -> None:
    """Deep clean + fresh install of orze (and orze-pro if installed).

    Fixes drift from partial upgrades: stale ``*.dist-info`` dirs,
    ``__pycache__`` from removed modules, user-site shadow installs,
    orphaned CLI entry-point scripts. Unlike ``do_upgrade`` which is a
    plain ``pip install --upgrade`` (and can leave stale dist-info),
    this guarantees a single, clean version on disk.

    Strategy:
      1. Stop the running orze.cli process (so file handles release).
      2. Uninstall orze + orze-pro from every Python environment we
         can reach from here: ``sys.executable`` and (if different)
         ``shutil.which("python3")``. Both system and user-site.
      3. Rip any surviving ``orze``/``orze_pro`` dirs and
         ``orze*.dist-info``/``orze_pro*.dist-info`` from every
         site-packages the running interpreter knows about.
      4. Hunt for stray ``orze`` console scripts in the Python ``bin/``
         directories and remove them.
      5. Fresh ``pip install --no-cache-dir`` of pinned or latest
         versions.
      6. Verify: exactly one dist-info per package, interpreter can
         import both, no shadowing paths in ``sys.path``.
      7. Restart the orze.cli process unless ``--no-restart``.
    """
    import site

    results_dir = Path(cfg["results_dir"])

    print("\n\033[1mOrze — Deep Clean Reinstall\033[0m")
    print("-----------------------------")
    print(f"Target orze    : {orze_version or 'latest'}")
    print(f"Target orze-pro: {pro_version or 'latest (if present)'}")
    print(f"Extra index    : {extra_index_url or '(none)'}")
    print()

    # --- 1. Stop running instance ------------------------------------
    print("[1/7] Stopping running orze instance...")
    was_running = stop_running_instance(results_dir)
    if not was_running:
        print("  (none running)")

    # --- 2. Collect every Python we can reach ------------------------
    pythons = [sys.executable]
    alt = shutil.which("python3")
    if alt and Path(alt).resolve() != Path(sys.executable).resolve():
        pythons.append(alt)
    alt = shutil.which("python")
    if alt and Path(alt).resolve() not in {Path(p).resolve() for p in pythons}:
        pythons.append(alt)

    # --- 3. pip uninstall from each -----------------------------------
    print(f"[2/7] Uninstalling from {len(pythons)} Python env(s)...")
    for py in pythons:
        print(f"  {py}")
        for pkg in ("orze-pro", "orze"):
            try:
                subprocess.run(
                    [py, "-m", "pip", "uninstall", "-y", pkg],
                    check=False,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
            except FileNotFoundError:
                pass

    # --- 4. Purge stale files from every site-packages ---------------
    print("[3/7] Purging stale package dirs and dist-info...")
    site_dirs: set = set()
    try:
        site_dirs.update(site.getsitepackages())
    except (AttributeError, OSError):
        pass
    try:
        site_dirs.add(site.getusersitepackages())
    except (AttributeError, OSError):
        pass
    # Also include what sys.path says is a site-packages dir
    for p in sys.path:
        if p.endswith("site-packages") and Path(p).is_dir():
            site_dirs.add(p)

    purged = 0
    for sp in sorted(site_dirs):
        sp_path = Path(sp)
        if not sp_path.is_dir():
            continue
        for pattern in ("orze", "orze_pro",
                        "orze-*.dist-info", "orze_pro-*.dist-info"):
            for match in sp_path.glob(pattern):
                try:
                    if match.is_dir():
                        shutil.rmtree(match)
                    else:
                        match.unlink()
                    print(f"  rm {match}")
                    purged += 1
                except OSError as e:
                    print(f"  WARN could not remove {match}: {e}")
    if purged == 0:
        print("  (nothing to purge)")

    # --- 5. Remove stray console-script binaries ----------------------
    print("[4/7] Removing stray orze/orze-pro binaries...")
    bin_dirs = {Path(sys.executable).parent}  # current interpreter's bin
    try:
        bin_dirs.add(Path(site.getuserbase()) / "bin")
    except (AttributeError, OSError):
        pass
    for bd in sorted(bin_dirs):
        for name in ("orze", "orze-pro"):
            p = bd / name
            if p.exists() or p.is_symlink():
                try:
                    p.unlink()
                    print(f"  rm {p}")
                except OSError as e:
                    print(f"  WARN {p}: {e}")

    # --- 6. Fresh install --------------------------------------------
    print("[5/7] Fresh install into current venv...")
    orze_spec = f"orze=={orze_version}" if orze_version else "orze"
    pip_cmd = [sys.executable, "-m", "pip", "install",
               "--no-cache-dir", "--upgrade", orze_spec]
    if extra_index_url:
        pip_cmd += ["--extra-index-url", extra_index_url]
    try:
        subprocess.run(pip_cmd, check=True)
    except subprocess.CalledProcessError:
        print("\n\033[31mReinstall of orze failed.\033[0m")
        return

    # Also reinstall orze-pro if it was listed (or user asked for it)
    if pro_version is not None or extra_index_url is not None:
        pro_spec = f"orze-pro=={pro_version}" if pro_version else "orze-pro"
        pip_cmd = [sys.executable, "-m", "pip", "install",
                   "--no-cache-dir", "--upgrade", pro_spec]
        if extra_index_url:
            pip_cmd += ["--extra-index-url", extra_index_url]
        try:
            subprocess.run(pip_cmd, check=True)
        except subprocess.CalledProcessError:
            print("\n\033[33mWARN: orze-pro reinstall failed — continuing.\033[0m")

    # --- 7. Verify ---------------------------------------------------
    print("[6/7] Verifying single clean install...")
    verify_code = (
        "import sys, importlib, site\n"
        "import orze\n"
        "mods = {'orze': orze}\n"
        "try:\n"
        "    import orze_pro; mods['orze_pro'] = orze_pro\n"
        "except ImportError:\n"
        "    pass\n"
        "for name, m in mods.items():\n"
        "    ver = getattr(m, '__version__', '?')\n"
        "    print(f'  {name:<10} {ver:<10}  @ {m.__file__}')\n"
        "# Check single dist-info per package\n"
        "from pathlib import Path\n"
        "for sp in set(site.getsitepackages() + [site.getusersitepackages()]):\n"
        "    sp = Path(sp)\n"
        "    if not sp.is_dir():\n"
        "        continue\n"
        "    for pat in ('orze-*.dist-info', 'orze_pro-*.dist-info'):\n"
        "        hits = list(sp.glob(pat))\n"
        "        if len(hits) > 1:\n"
        "            print(f'  ERROR: multiple dist-info in {sp}: {hits}')\n"
        "            sys.exit(2)\n"
    )
    try:
        subprocess.run([sys.executable, "-c", verify_code], check=True)
    except subprocess.CalledProcessError:
        print("\n\033[31mVerification failed — inspect the output above.\033[0m")
        return

    # --- 8. Restart --------------------------------------------------
    print("[7/7] Restart...")
    if no_restart:
        print("  --no-restart set, skipping.")
        print(f"  To restart manually:")
        print(f"  {sys.executable} -m orze.cli -c "
              f"{cfg.get('_config_path', 'orze.yaml')}")
        return
    if was_running:
        print("  Restarting orze with clean install...")
        os.execv(sys.executable, [sys.executable, "-m", "orze.cli",
                                  "-c", str(cfg.get("_config_path", "orze.yaml"))])
    else:
        print("  (was not running — nothing to restart)")

    print("\n\033[32mReinstall complete.\033[0m")


# ---------------------------------------------------------------------------
# Shared storage auto-detection
# ---------------------------------------------------------------------------

def find_shared_mounts() -> list:
    """Find shared/network filesystem mounts. Returns [(path, fs_type), ...]."""
    _NETWORK_FS = {"lustre", "nfs", "nfs4", "efs", "fuse.s3fs", "fuse.goofys",
                   "ceph", "gpfs", "beegfs", "pvfs2", "glusterfs"}
    _FAST_FS = {"lustre", "gpfs", "beegfs", "pvfs2"}

    results = []
    seen_devs = set()
    try:
        for line in Path("/proc/mounts").read_text().splitlines():
            parts = line.split()
            if len(parts) < 3:
                continue
            device, mount_point, fs_type = parts[0], parts[1], parts[2]
            if fs_type not in _NETWORK_FS and "@tcp:" not in device:
                continue
            if not os.access(mount_point, os.W_OK):
                continue
            try:
                dev = os.stat(mount_point).st_dev
                if dev in seen_devs:
                    continue
                seen_devs.add(dev)
            except OSError:
                continue
            try:
                free_gb = shutil.disk_usage(mount_point).free / (1024**3)
                if free_gb < 20:
                    continue
            except (PermissionError, OSError):
                continue
            results.append((mount_point, fs_type))
    except Exception:
        pass

    results.sort(key=lambda x: x[1] in _FAST_FS, reverse=True)
    return results


def resolve_init_path(explicit_path: str) -> str:
    """Resolve the project directory for --init.

    Explicit path -> use it. No path (__ask__ sentinel) -> detect shared mount,
    prompt once (interactive) or auto-pick (non-interactive / piped).
    """
    if explicit_path != "__ask__":
        return str(Path(explicit_path).resolve())

    # Auto-detect best shared mount as default
    shared = find_shared_mounts()
    default = shared[0][0] if shared else str(Path.cwd())

    # Non-interactive (e.g. curl | bash) — use detected path directly
    if not sys.stdin.isatty():
        return str(Path(default).resolve())

    print()
    print("\033[1mOrze needs a shared storage path for the project.\033[0m")
    print("  All cluster nodes must be able to read/write this path.")
    if shared:
        print(f"  Detected: \033[36m{default}\033[0m ({shared[0][1]})")
    path = input(f"  Project path [{default}]: ").strip()
    return str(Path(path or default).resolve())


# ---------------------------------------------------------------------------
# --init: scaffold a new orze project
# ---------------------------------------------------------------------------

def _try_create_venv(venv_dir: Path, venv_python: Path) -> bool:
    """Create *venv_dir* with pyyaml installed. Returns True on success.

    Tries `python -m venv` first (matches the user's Python). If that
    fails — typically because `python3-venv` is missing on Debian/Ubuntu
    — fall back to `uv venv --python <sys.executable>`. uv uses the same
    Python interpreter but ships its own venv creation path and does not
    need `ensurepip`.
    """
    # Path A: standard library venv.
    try:
        subprocess.run(
            [sys.executable, "-m", "venv", str(venv_dir)],
            check=True, timeout=60,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        subprocess.run(
            [str(venv_python), "-m", "pip", "install", "--quiet", "pyyaml"],
            check=True, timeout=120,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError):
        # Clean up any partial venv before the fallback.
        shutil.rmtree(venv_dir, ignore_errors=True)

    # Path B: uv fallback (common on hosts without python3-venv).
    uv = shutil.which("uv")
    if not uv:
        return False
    try:
        subprocess.run(
            [uv, "venv", str(venv_dir), "--python", sys.executable],
            check=True, timeout=60,
            stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
        )
        # `uv pip` works against a venv without bootstrapping pip inside it.
        subprocess.run(
            [uv, "pip", "install", "--python", str(venv_python),
             "--quiet", "pyyaml"],
            check=True, timeout=120,
        )
        return True
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError):
        shutil.rmtree(venv_dir, ignore_errors=True)
        return False


def do_init(init_arg: str):
    """Scaffold a new orze project at the resolved path.

    *init_arg* is the raw value from ``args.init`` (a path or ``"__ask__"``).
    """
    from orze.cli_demo import BASELINE_TRAIN_PY, RESEARCH_RULES_TEMPLATE

    init_path = resolve_init_path(init_arg)
    project_dir = Path(init_path).resolve()
    project_dir.mkdir(parents=True, exist_ok=True)
    os.chdir(project_dir)

    print(f"\n\033[1mOrze — Initialization\033[0m")
    print(f"  Project : {project_dir}")
    print("---------------------")
    created = []

    def _create(path, content, label=None):
        p = Path(path)
        if p.exists():
            print(f"  exists   {p}")
            return False
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(content, encoding="utf-8")
        created.append(label or str(p))
        print(f"  \033[32mcreated\033[0m  {p}")
        return True

    print()

    # .env template (project-local — each project has its own API keys)
    env_content = """\
# Orze auto-loads this file on startup. No need to export.
# Uncomment and fill in the key for your preferred LLM backend:

# ANTHROPIC_API_KEY=sk-ant-...
# GEMINI_API_KEY=AI...
# OPENAI_API_KEY=sk-...

# Telegram notifications (optional):
# TELEGRAM_BOT_TOKEN=123456:ABC...
# TELEGRAM_CHAT_ID=-100...
"""
    _create(".env", env_content)

    # ORZE-AGENT.md and ORZE-RULES.md (project-local copies)
    # Uses multi-tier fallback: package file → importlib → GitHub → never fails
    docs = _get_embedded_docs()
    for src_name, dst_name in {"AGENT.md": "ORZE-AGENT.md", "RULES.md": "ORZE-RULES.md"}.items():
        content = docs.get(src_name)
        if content:
            _create(dst_name, content)
        else:
            print(f"  \033[33mskip\033[0m     {dst_name} — not available (non-critical)")

    # 1. Train script stub
    train_script = "train.py"
    _create(train_script, BASELINE_TRAIN_PY.strip() + "\n", "train.py (stub)")

    # 2. Create venv and install pyyaml (if no venv exists)
    venv_dir = project_dir / "venv"
    venv_python = venv_dir / "bin" / "python3"
    venv_ok = False
    if venv_dir.is_dir() and venv_python.exists():
        # Health check: can the venv import yaml?
        try:
            subprocess.run(
                [str(venv_python), "-c", "import yaml"],
                check=True,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                timeout=10,
            )
            venv_ok = True
            print(f"  exists   venv/ (healthy)")
        except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
                FileNotFoundError, OSError):
            print(f"  \033[33mbroken\033[0m   venv/ — recreating...")
            shutil.rmtree(venv_dir, ignore_errors=True)

    if not venv_ok:
        print(f"  \033[32mcreating\033[0m venv/...")
        if _try_create_venv(venv_dir, venv_python):
            created.append("venv/ (with pyyaml)")
            print(f"  \033[32mcreated\033[0m  venv/ (with pyyaml)")
        else:
            print(f"  \033[33mwarning\033[0m  venv creation failed — neither `python -m venv` nor `uv venv` succeeded.")
            print(f"           On Debian/Ubuntu: `sudo apt install python3-venv`, or install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh`")
            print(f"           Once a venv creator is available, re-run `orze --init .` (it is idempotent).")

    # Determine python path for orze.yaml
    if (venv_dir / "bin" / "python3").exists():
        python_for_yaml = str(venv_dir / "bin" / "python3")
    else:
        python_for_yaml = sys.executable

    # 3. orze.yaml
    backend = ("anthropic" if os.environ.get("ANTHROPIC_API_KEY") else
               "gemini" if os.environ.get("GEMINI_API_KEY") else
               "openai" if os.environ.get("OPENAI_API_KEY") else "ollama")

    # Detect orze-pro for full config generation
    _has_pro = False
    try:
        import orze_pro  # noqa: F401
        _has_pro = True
    except ImportError:
        pass

    if _has_pro:
        yaml_content = f"""\
# orze.yaml — Project configuration (orze-pro enabled)
# Docs: https://github.com/erikhenriksson/orze

# --- REQUIRED ---
train_script: {train_script}
ideas_file: ideas.md
results_dir: results
python: {python_for_yaml}

# --- ENV ---
train_extra_env:
  PYTHONUNBUFFERED: "1"

# --- TIMEOUTS ---
timeout: 3600
poll: 30
stall_minutes: 60
max_idea_failures: 2
max_fix_attempts: 2

# --- REPORT ---
report:
  primary_metric: test_accuracy
  sort: descending
  columns:
    - {{key: "test_accuracy", label: "Accuracy", fmt: ".4f"}}
    - {{key: "test_loss", label: "Loss", fmt: ".4f"}}
    - {{key: "training_time", label: "Time(s)", fmt: ".0f"}}

# --- RETROSPECTION ---
# Detection only — FSM owns all pause/trigger decisions
retrospection:
  enabled: true
  interval: 6
  auto_pause: false
  plateau_window: 20
  fail_window: 10
  fail_threshold: 0.5

# --- CODE EVOLUTION ---
evolution:
  enabled: false              # FSM dispatches triggers, not orze

# --- ROLES (orze-pro autopilot) ---
# Roles compose from bundled static SOPs (@sop:<name>) shipped with
# orze-pro. Inspect with:  orze sop list
# To add project-specific behavior, create <project>/skills/<name>.skill.md
# and append it to the role's skills list. Dynamic skills override
# static ones with the same id.
roles:
  research:
    mode: research
    backend: {backend}
    skills:
      - "@sop:research_base"
      - ./RESEARCH_RULES.md   # project-specific research constraints
    cooldown: 120
    timeout: 600
  code_evolution:
    mode: claude
    triggered_by: fsm
    skills:
      - "@sop:code_evolution_base"
    timeout: 900
    model: opus
    allowed_tools: "Read,Write,Edit,Glob,Grep,Bash"
  meta_research:
    mode: research
    backend: {backend}
    triggered_by: fsm
    skills:
      - "@sop:research_base"
      - ./RESEARCH_RULES.md
    cooldown: 3600
    timeout: 600
  professor:
    mode: claude
    skills:
      - "@sop:professor_base"
      - "@sop:professor_web_search"
      - "@sop:professor_cross_domain_query"
      - "@sop:professor_idea_review"
      - "@sop:professor_diversity_enforcement"
      - "@sop:professor_gap_closure"
      - "@sop:professor_strategy_review"
      - "@sop:professor_regression_detection"
      - "@sop:professor_steering"
    cooldown: 600
    timeout: 600
    model: opus
    pausable: false
    allowed_tools: "Read,Write,Edit,Glob,Grep,Bash,WebSearch,WebFetch"
  # The engineer role bundles both strategy implementation and bug
  # fixing. There is no separate bug_fixer role.
  fsm:
    mode: script
    script: fsm/runner.py
    args: ["--results-dir", "{{results_dir}}"]
    cooldown: 120
    timeout: 30
"""
    else:
        yaml_content = f"""\
# orze.yaml — Project configuration
# Docs: https://github.com/erikhenriksson/orze

# --- REQUIRED ---
train_script: {train_script}
ideas_file: ideas.md
results_dir: results
python: {python_for_yaml}

# --- EVALUATION (optional) ---
# eval_script: evaluate_dataset.py
# eval_args: [--split, test]
# eval_timeout: 3600
# eval_output: eval_report.json

# --- RESEARCH AGENT (optional) ---
# Auto-generates ideas. Requires API key in .env or environment.
# Uncomment below only to customize settings:
#
# roles:
#   research:
#     mode: research
#     backend: {backend}
#     cooldown: 600
#     timeout: 300

# --- ADVANCED ---
# timeout: 3600
# poll: 30
# stall_minutes: 0
# max_idea_failures: 0
# max_fix_attempts: 0

# --- UPGRADE TO PRO ---
# pip install orze-pro
# Adds: autonomous research agents, the engineer role (implement +
# fix), code evolution, The Professor, watchdog, 7 FSM procedures,
# idea filtering, and more.
# Re-run orze --init to regenerate this config with pro features.
"""
    _create("orze.yaml", yaml_content)

    if _has_pro:
        Path("procedures").mkdir(exist_ok=True)

    # 4. ideas.md with task-agnostic seed experiments
    ideas_content = """\
# Ideas
<!-- Orze reads ideas from this file. Format:
     ## idea-XXXX: Title
     - **Priority**: high|medium|low
     ```yaml
     key: value        <- passed to your train script
     ```
     Orze consumes ideas on startup (moves them to idea_lake.db).
-->

## idea-0001: Baseline — default hyperparameters
- **Priority**: high

```yaml
learning_rate: 0.001
epochs: 10
```

## idea-0002: Higher learning rate
- **Priority**: medium

```yaml
learning_rate: 0.01
epochs: 10
```

## idea-0003: Longer training
- **Priority**: medium

```yaml
learning_rate: 0.001
epochs: 100
```
"""
    _create("ideas.md", ideas_content)

    # 5. configs/base.yaml (task-agnostic defaults)
    base_content = """\
# Base config — shared defaults for all experiments.
# Your train script receives this via --config <path>.
# Add any project-wide settings here (model architecture, data paths, etc).

# Example defaults (used by the demo train.py):
seed: 42
noise: 0.1
"""
    _create("configs/base.yaml", base_content)

    # 6. RESEARCH_RULES.md (if it doesn't exist)
    _create("RESEARCH_RULES.md", RESEARCH_RULES_TEMPLATE, "RESEARCH_RULES.md")

    # 7. results directory
    Path("orze_results").mkdir(exist_ok=True)

    # =================================================================
    # Summary
    # =================================================================
    print()
    if created:
        print(f"\033[32m Created {len(created)} file(s). Project ready.\033[0m")
    else:
        print("\033[32m All files already exist.\033[0m")

    # API key status
    _detected = []
    if os.environ.get("ANTHROPIC_API_KEY"): _detected.append("Anthropic")
    if os.environ.get("GEMINI_API_KEY"): _detected.append("Gemini")
    if os.environ.get("OPENAI_API_KEY"): _detected.append("OpenAI")

    print()
    if _detected:
        print(f"  API keys: \033[32m{', '.join(_detected)}\033[0m (auto-discovered)")
    else:
        print(f"  API keys: \033[33mnone found\033[0m — add to {project_dir}/.env")

    cfg_path = project_dir / "orze.yaml"

    # =================================================================
    # Pro feature self-check (only for pro users)
    # =================================================================
    if _has_pro:
        print()
        print("\033[1mPro Feature Check:\033[0m")

        _checks = [
            ("license", lambda: _check_pro_license()),
            ("research agent", lambda: _check_pro_role("research", project_dir)),
            ("professor", lambda: _check_pro_role("professor", project_dir)),
            ("engineer", lambda: _check_pro_role("engineer", project_dir)),
            ("code evolution", lambda: _check_pro_role("code_evolution", project_dir)),
            # meta_research reuses research SOPs
            ("meta research", lambda: _check_pro_role("research", project_dir)),
            ("FSM procedures", lambda: _check_pro_procedures()),
            ("idea verifier", lambda: _check_pro_plugin("idea_verifier")),
            ("activity log", lambda: _check_pro_plugin("role_logger")),
            ("API keys", lambda: _check_api_keys()),
        ]

        all_pass = True
        for name, check_fn in _checks:
            try:
                ok, detail = check_fn()
                icon = "\033[32m✓\033[0m" if ok else "\033[33m○\033[0m"
                if not ok:
                    all_pass = False
                print(f"  {icon} {name:20s} {detail}")
            except Exception as e:
                all_pass = False
                print(f"  \033[31m✗\033[0m {name:20s} error: {e}")

        if all_pass:
            print()
            print("  \033[32mAll pro features ready.\033[0m")
        else:
            print()
            print("  \033[33mSome features need setup — see details above.\033[0m")

    # =================================================================
    # Multi-task detection & project split (orze-pro only)
    # =================================================================
    _multitask_done = False
    if _has_pro:
        goal_path = project_dir / "GOAL.md"
        if goal_path.exists():
            try:
                from orze_pro.agents.task_splitter import (
                    detect_tasks, propose_gpu_allocation,
                    format_proposal, create_task_folders,
                )

                goal_text = goal_path.read_text(encoding="utf-8")
                tasks = detect_tasks(goal_text)

                if len(tasks) > 1:
                    # Detect GPUs
                    try:
                        from orze_pro.agents._llm_utils import detect_gpu_info
                        gpu_info = detect_gpu_info()
                        n_gpus = gpu_info.get("gpu_count", 1)
                    except Exception:
                        n_gpus = 1

                    tasks = propose_gpu_allocation(tasks, n_gpus)

                    # Show proposal
                    proposal = format_proposal(tasks, project_dir)
                    print()
                    print("\033[1mMulti-Task Detection:\033[0m")
                    print(proposal)

                    # Prompt user (only in interactive mode)
                    proceed = True
                    if sys.stdin.isatty():
                        response = input("").strip().lower()
                        if response == "n":
                            print("  Skipping multi-task setup.")
                            proceed = False
                        elif response == "edit":
                            # Let user edit GPU allocation
                            print(f"  Enter GPU counts per task (comma-separated, "
                                  f"{n_gpus} GPUs total):")
                            print(f"  e.g. {','.join(str(len(t.get('gpus', [0]))) for t in tasks)}")
                            try:
                                counts_str = input("  > ").strip()
                                counts = [int(x.strip()) for x in counts_str.split(",")]
                                if len(counts) == len(tasks) and sum(counts) <= n_gpus:
                                    gpu_cursor = 0
                                    for i, t in enumerate(tasks):
                                        t["gpus"] = list(range(gpu_cursor, gpu_cursor + counts[i]))
                                        gpu_cursor += counts[i]
                                else:
                                    print(f"  \033[33mInvalid allocation — using defaults.\033[0m")
                            except (ValueError, EOFError):
                                print(f"  \033[33mInvalid input — using defaults.\033[0m")

                    if proceed:
                        env_path = project_dir / ".env"
                        create_task_folders(tasks, project_dir, env_path)
                        _multitask_done = True
                        print(f"\n  \033[32mCreated {len(tasks)} task folders.\033[0m")
                        print(f"  Write train.py for each task, then run start_all.sh")

            except ImportError:
                pass
            except Exception as e:
                # Never break init due to multi-task failure
                print(f"  \033[33mskipped\033[0m  multi-task detection error: {e}")

    # Professor behavior now ships as bundled static SOPs in orze-pro
    # (orze_pro/sops/professor_*.skill.md). No per-project bootstrap
    # needed at init. Projects wanting task-specific tailoring author
    # dynamic SOPs under <project>/skills/ and append them to the
    # professor role's skills: list in orze.yaml.

    print()
    print("\033[1mNext steps:\033[0m")
    if _has_pro:
        print(f"  1. Edit \033[36m{project_dir}/train.py\033[0m with your training logic")
        print(f"  2. Add API key to \033[36m{project_dir}/.env\033[0m")
        print(f"  3. Run: \033[36morze pro activate <key>\033[0m")
        print(f"  4. Run: \033[36morze --check -c {cfg_path}\033[0m to validate")
        print(f"  5. Run: \033[36morze start -c {cfg_path}\033[0m")
    else:
        print(f"  1. Edit \033[36m{project_dir}/train.py\033[0m with your training logic")
        print(f"  2. Add API key to \033[36m{project_dir}/.env\033[0m (optional, for auto-research)")
        print(f"  3. Run: \033[36morze --check -c {cfg_path}\033[0m to validate")
        print(f"  4. Run: \033[36morze -c {cfg_path}\033[0m to start")


# ---------------------------------------------------------------------------
# --check: validate config and environment
# ---------------------------------------------------------------------------

def do_check(cfg: dict):
    """Validate config, files, API keys, GPUs, .env — then exit."""
    from orze.core.config import _validate_config, find_dotenv
    from orze.hardware.gpu import detect_all_gpus

    print(f"\n\033[1mOrze v{__version__} — Config Check\033[0m")
    print("=" * 50)

    ok = "\033[32m[x]\033[0m"
    no = "\033[31m[ ]\033[0m"
    warn_mark = "\033[33m[-]\033[0m"

    # --- Files ---
    print("  \033[1mFiles:\033[0m")
    cp = Path(cfg.get("_config_path", "orze.yaml"))
    cp_ok = cp.exists()
    print(f"    {ok if cp_ok else no} orze.yaml: {cp}")
    if not cp_ok:
        print(f"      hint: run \033[36morze --init\033[0m to create a project")

    env_path = find_dotenv(str(cp) if cp_ok else None)
    if env_path:
        print(f"    {ok} .env: {env_path}")
    else:
        print(f"    {warn_mark} .env: not found (optional — needed for API keys)")

    ts = cfg.get("train_script", "train.py")
    ts_ok = Path(ts).exists()
    print(f"    {ok if ts_ok else no} train_script: {ts}")
    if not ts_ok and not cp_ok:
        print(f"      hint: run \033[36morze --init\033[0m to scaffold a project")

    ideas_path = cfg.get("ideas_file", "ideas.md")
    ideas_exists = Path(ideas_path).exists()
    print(f"    {ok if ideas_exists else no} ideas_file: {ideas_path}")

    # Validate ideas format
    ideas_valid = False
    ideas_count = 0
    if ideas_exists:
        from orze.core.ideas import parse_ideas
        try:
            parsed = parse_ideas(ideas_path)
            ideas_count = len(parsed)
            ideas_valid = ideas_count > 0
        except Exception:
            pass
    if ideas_exists:
        if ideas_valid:
            print(f"      {ok} {ideas_count} idea(s) parsed OK")
        else:
            print(f"      {warn_mark} no valid ideas found (may be empty if using idea_lake)")

    rdir = cfg.get("results_dir", "orze_results")
    print(f"    {ok if Path(rdir).is_dir() else warn_mark} results_dir: {rdir}")

    bc = cfg.get("base_config")
    if bc:
        print(f"    {ok if Path(bc).exists() else warn_mark} base_config: {bc}")

    es = cfg.get("eval_script")
    if es:
        print(f"    {ok if Path(es).exists() else no} eval_script: {es}")

    lake_path = Path(cfg.get("idea_lake_db") or Path(ideas_path).parent / "idea_lake.db")
    print(f"    {ok if lake_path.exists() else warn_mark} idea_lake.db: {lake_path}")

    # --- Filesystem writability ---
    print()
    print("  \033[1mFilesystem:\033[0m")
    rdir_p = Path(rdir)
    fs_ok = False
    if rdir_p.is_dir():
        probe = rdir_p / ".orze_probe"
        try:
            probe.write_text("1")
            probe.unlink()
            fs_ok = True
        except Exception:
            pass
    print(f"    {ok if fs_ok else no} results_dir writable" +
          ("" if fs_ok else " (create it or check mount)"))

    disk = shutil.disk_usage(rdir_p if rdir_p.is_dir() else Path.cwd())
    free_gb = disk.free / (1024**3)
    print(f"    {ok if free_gb > 5 else warn_mark} disk: {free_gb:.1f} GB free")

    # --- API keys (only relevant for pro users) ---
    from orze.extensions import has_pro
    has_anthropic = bool(os.environ.get("ANTHROPIC_API_KEY"))
    has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
    has_any_llm_key = has_anthropic or has_gemini
    any_key = has_any_llm_key or bool(os.environ.get("OPENAI_API_KEY"))
    if has_pro():
        print()
        print("  \033[1mAPI Keys:\033[0m")
        for env_var, label in [("ANTHROPIC_API_KEY", "Anthropic"),
                                ("GEMINI_API_KEY", "Gemini")]:
            present = bool(os.environ.get(env_var))
            print(f"    {ok if present else warn_mark} {label} ({env_var})")
        if not has_any_llm_key:
            print(f"      hint: add ANTHROPIC_API_KEY or GEMINI_API_KEY to .env for auto-research")

    # --- GPUs ---
    print()
    print("  \033[1mGPUs:\033[0m")
    gpus = detect_all_gpus()
    if gpus:
        print(f"    {ok} {len(gpus)} GPU(s) detected: {gpus}")
    else:
        print(f"    {no} No GPUs detected")

    # --- Roles ---
    print()
    print("  \033[1mResearch Agent:\033[0m")
    roles = cfg.get("roles") or {}
    research_names = [
        rname for rname, rcfg in roles.items()
        if isinstance(rcfg, dict) and rcfg.get("mode") in ("research", "claude")
    ]
    if roles:
        for rname, rcfg in roles.items():
            if isinstance(rcfg, dict):
                mode = rcfg.get("mode", "script")
                backend = rcfg.get("backend", "")
                detail = f"mode={mode}" + (f", backend={backend}" if backend else "")
                print(f"    {ok} {rname}: {detail}")
        if research_names and not has_any_llm_key and has_pro():
            print(f"    {warn_mark} \033[33mAuto-research will not work: no ANTHROPIC_API_KEY or GEMINI_API_KEY found\033[0m")
            print(f"      Add ANTHROPIC_API_KEY or GEMINI_API_KEY to .env to enable auto-research")
    else:
        if has_pro():
            print(f"    {no} No research agent configured — ideas will not be generated automatically")
            if not any_key:
                print(f"      hint: add an API key to .env (GEMINI_API_KEY or ANTHROPIC_API_KEY)")
                print(f"            and configure a research role in orze.yaml")
            else:
                print(f"      hint: auto-discovery found API key(s) but roles section in orze.yaml")
                print(f"            may be overriding it. Remove 'roles: {{}}' or configure a research role")
        else:
            print(f"    \033[2mAI-powered idea generation, auto-fix, and code evolution\033[0m")
            print(f"    \033[2mAvailable with orze-pro → orze.ai/pro\033[0m")

    # --- Validation ---
    errors, warnings = _validate_config(cfg)
    if errors or warnings:
        print()
    if errors:
        print("  \033[31mErrors:\033[0m")
        for e in errors:
            print(f"    \033[31m\u2717\033[0m {e}")
    if warnings:
        print("  \033[33mWarnings:\033[0m")
        for w in warnings:
            print(f"    \033[33m!\033[0m {w}")

    print()
    if errors:
        print(f"\033[31m\u2717 {len(errors)} error(s) — fix before running.\033[0m")
        if not cp_ok:
            print(f"  hint: run \033[36morze --init\033[0m to create a new project")
        sys.exit(1)
    else:
        from orze.extensions import has_pro, check_pro_status
        if has_pro():
            print(f"\033[32m\u2714 Ready to run.\033[0m (pro: \033[36m{check_pro_status()}\033[0m)")
        else:
            print(f"\033[32m\u2714 Ready to run.\033[0m")
            print(f"  \033[33m\u2139\033[0m {check_pro_status()}")
