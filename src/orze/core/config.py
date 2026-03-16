import os
import logging
import copy
from typing import Optional
from pathlib import Path
import yaml

logger = logging.getLogger("orze")

import sys

def find_dotenv(config_path: Optional[str] = None) -> Optional[Path]:
    """Find .env file: next to config or CWD. Returns path or None."""
    candidates = []
    if config_path:
        candidates.append(Path(config_path).resolve().parent / ".env")
    candidates.append(Path.cwd() / ".env")
    for c in candidates:
        if c.is_file():
            return c
    return None


def _load_dotenv(config_path: Optional[str] = None) -> int:
    """Load .env file. Only sets vars NOT already in os.environ. Returns count loaded."""
    env_file = find_dotenv(config_path)

    if not env_file:
        return 0

    loaded = 0
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:]
        key, _, value = line.partition("=")
        key = key.strip()
        value = value.strip().strip("'\"")
        if key and not os.environ.get(key):
            os.environ[key] = value
            loaded += 1

    if loaded:
        logger.info("Loaded %d env var(s) from %s", loaded, env_file)
    return loaded


DEFAULT_CONFIG = {
    "train_script": "train.py",
    "ideas_file": "ideas.md",
    "base_config": "configs/base.yaml",
    "results_dir": "results",
    "python": sys.executable,
    "train_extra_args": [],
    "train_extra_env": {},
    "timeout": 3600,
    "poll": 30,
    "gpu_mem_threshold": 2000,
    "pre_script": None,
    "pre_args": [],
    "pre_timeout": 3600,
    "eval_script": None,
    "eval_args": [],
    "eval_timeout": 3600,
    "eval_output": "eval_report.json",
    "post_scripts": [],
    "cleanup": {
        "script": None,
        "interval": 100,
        "patterns": [],
    },
    "report": {
        "title": "Orze Report",
        "primary_metric": "test_accuracy",
        "sort": "descending",
        "columns": [
            {"key": "test_accuracy", "label": "Accuracy", "fmt": ".4f"},
            {"key": "test_loss", "label": "Loss", "fmt": ".4f"},
            {"key": "training_time", "label": "Time(s)", "fmt": ".0f"},
        ],
        "ceiling_k": 20,
        "ceiling_std_threshold": 0.015,
        "ceiling_min_ideas": 30,
    },
    "stall_minutes": 0,         # 0 = disabled
    "max_idea_failures": 0,     # 0 = disabled (never skip)
    "max_fix_attempts": 0,      # 0 = disabled; executor LLM fix attempts per idea
    "min_disk_gb": 0,           # 0 = disabled
    "orphan_timeout_hours": 0,  # 0 = disabled
    "plateau_threshold": 50,    # fire plateau notification after N completions w/o improvement
    "roles": {},
    "auto_upgrade": False,
    "notifications": {
        "enabled": False,
        "on": ["completed", "failed", "new_best", "watchdog_restart", "plateau"],
        "channels": [],
    },
    "retrospection": {
        "enabled": False,
        "script": "",
        "interval": 50,
        "timeout": 120,
    },
}
def load_project_config(path: Optional[str] = None) -> dict:
    """Load orze.yaml and merge with defaults. Returns full config dict."""
    _load_dotenv(path)
    cfg = copy.deepcopy(DEFAULT_CONFIG)

    if not path and Path("orze.yaml").exists():
        path = "orze.yaml"

    if path and Path(path).exists():
        raw = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
        for k, v in raw.items():
            if k == "report" and isinstance(v, dict):
                cfg["report"] = {**cfg["report"], **v}
            else:
                cfg[k] = v
        # Fix YAML 'on:' boolean parsing — YAML interprets 'on' as True
        # so notifications.on becomes notifications[True] instead of notifications["on"]
        ncfg = cfg.get("notifications")
        if isinstance(ncfg, dict) and True in ncfg and "on" not in ncfg:
            ncfg["on"] = ncfg.pop(True)
            logger.info("Fixed YAML 'on:' boolean key in notifications config")

        logger.info("Loaded config from %s", path)
    elif path:
        logger.warning("Config file %s not found, using defaults", path)

    # Migrate legacy research: into roles: dict
    if "research" in cfg and isinstance(cfg["research"], dict):
        logger.warning("Migrating legacy 'research:' config to 'roles: {research: ...}'. "
                        "Update orze.yaml to use the 'roles:' format directly.")
        if not cfg.get("roles"):
            cfg["roles"] = {"research": cfg["research"]}
        elif "research" not in cfg["roles"]:
            cfg["roles"]["research"] = cfg["research"]

    # Auto-discover research backends from environment API keys.
    # Only activates if no research roles are explicitly configured.
    roles = cfg.get("roles") or {}
    has_research_role = any(
        isinstance(rc, dict) and rc.get("mode") in ("claude", "research")
        for rc in roles.values()
    )
    if not has_research_role:
        _AUTO_BACKENDS = [
            ("GEMINI_API_KEY", "gemini", "gemini-2.5-flash"),
            ("OPENAI_API_KEY", "openai", "gpt-4o"),
            ("ANTHROPIC_API_KEY", "anthropic", None),
        ]
        discovered = []
        for env_var, backend, default_model in _AUTO_BACKENDS:
            if os.environ.get(env_var):
                role_name = f"research_{backend}"
                role_cfg = {"mode": "research", "backend": backend}
                if default_model:
                    role_cfg["model"] = default_model
                if "roles" not in cfg:
                    cfg["roles"] = {}
                cfg["roles"][role_name] = role_cfg
                discovered.append(f"{backend} ({env_var})")
        if discovered:
            logger.info("Auto-discovered research backends: %s",
                        ", ".join(discovered))

    return cfg


def _validate_config(cfg: dict) -> tuple:
    """Validate orze config on startup. Returns (errors, warnings) tuple."""
    errors = []
    warnings = []

    # --- Errors: things that will break ---

    # Validate roles
    roles = cfg.get("roles")
    if roles and isinstance(roles, dict):
        for rname, rcfg in roles.items():
            if not isinstance(rcfg, dict):
                errors.append(f"roles.{rname}: expected dict, got {type(rcfg).__name__}")
                continue
            mode = rcfg.get("mode", "script")
            if mode not in ("script", "claude", "research"):
                errors.append(f"roles.{rname}.mode: '{mode}' is invalid "
                              f"(expected 'script', 'claude', or 'research')")
            if mode == "claude" and not rcfg.get("rules_file"):
                errors.append(f"roles.{rname}: mode 'claude' requires 'rules_file'")
            if mode == "script" and not rcfg.get("script"):
                errors.append(f"roles.{rname}: mode 'script' requires 'script'")
            if mode == "research" and not rcfg.get("backend"):
                errors.append(f"roles.{rname}: mode 'research' requires 'backend' "
                              f"(gemini, openai, anthropic, ollama, custom)")

    # Validate numeric fields
    for key in ("timeout", "poll", "eval_timeout", "stall_minutes",
                "max_idea_failures", "max_fix_attempts", "min_disk_gb",
                "orphan_timeout_hours"):
        val = cfg.get(key)
        if val is not None and (not isinstance(val, (int, float)) or val < 0):
            errors.append(f"{key}: must be a non-negative number, got {val!r}")

    # Validate eval config consistency
    if cfg.get("eval_script") and not cfg.get("eval_output"):
        errors.append("eval_script is set but eval_output is missing")

    # train_script must exist
    ts = cfg.get("train_script")
    if ts and not Path(ts).exists():
        errors.append(f"train_script not found: {ts}")

    # --- Warnings: things that might be unintentional ---

    bc = cfg.get("base_config")
    if bc and not Path(bc).exists():
        warnings.append(f"base_config not found: {bc}")

    es = cfg.get("eval_script")
    if es and not Path(es).exists():
        warnings.append(f"eval_script not found: {es}")

    if not roles:
        warnings.append("No roles configured — idea generation disabled")

    # Check for API keys if research roles exist
    has_research = roles and any(
        isinstance(rc, dict) and rc.get("mode") == "research"
        for rc in roles.values()
    )
    if has_research:
        has_key = any(os.environ.get(k) for k in
                      ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"))
        if not has_key:
            warnings.append("Research role configured but no API keys found in environment")

    gc_cfg = cfg.get("gc") or {}
    if not gc_cfg.get("enabled"):
        warnings.append("GC disabled — checkpoint dirs will accumulate indefinitely. "
                        "Add gc: {enabled: true, checkpoints_dir: ...} to enable.")

    ncfg = cfg.get("notifications", {})
    if not ncfg.get("enabled"):
        warnings.append("Notifications disabled")
    else:
        channels = ncfg.get("channels", [])
        if not channels:
            warnings.append("Notifications enabled but no channels configured")
        for ch in channels:
            if not isinstance(ch, dict):
                continue
            ch_type = ch.get("type", "?")
            if ch_type == "telegram":
                if not ch.get("bot_token"):
                    warnings.append(f"Notification channel '{ch_type}': missing bot_token")
                if not ch.get("chat_id"):
                    warnings.append(f"Notification channel '{ch_type}': missing chat_id")
            elif ch_type in ("slack", "discord"):
                if not ch.get("webhook_url"):
                    warnings.append(f"Notification channel '{ch_type}': missing webhook_url")
            elif ch_type == "webhook":
                if not ch.get("url"):
                    warnings.append(f"Notification channel '{ch_type}': missing url")

    return errors, warnings

def _sanitize_config(config: dict) -> dict:
    """Sanitize config by replacing invalid numeric values with defaults.

    Handles cases where AI-generated ideas use strings like 'variable' or 'auto'
    in fields that expect integers (e.g., sequence_length, max_frames).
    """
    if not isinstance(config, dict):
        return config

    # Known numeric fields that should be integers
    numeric_fields = {
        ("training", "sequence_length"): 32,
        ("training", "batch_size"): 16,
        ("training", "epochs"): 10,
        ("data", "batch_size"): 16,
        ("data", "frame_sampling", "max_frames"): 32,
        ("optimizer", "max_epochs"): 10,
    }

    # Known intermediate fields that must be dicts (not lists/scalars)
    dict_fields = {
        ("data", "frame_sampling"): {},
    }

    def sanitize_value(value, default):
        """Try to convert to int; if it fails, return default."""
        if isinstance(value, int):
            return value
        if isinstance(value, str):
            try:
                return int(value)
            except ValueError:
                logger.warning(f"Replacing invalid numeric value '{value}' with {default}")
                return default
        return value

    # Deep copy to avoid modifying original
    config = copy.deepcopy(config)

    # Ensure known intermediate fields are dicts
    for path, default in dict_fields.items():
        current = config
        for key in path[:-1]:
            if key not in current or not isinstance(current[key], dict):
                break
            current = current[key]
        else:
            final_key = path[-1]
            if final_key in current and not isinstance(current[final_key], dict):
                logger.warning(
                    f"Replacing non-dict value for '{'.'.join(path)}' "
                    f"(was {type(current[final_key]).__name__}) with {default}"
                )
                current[final_key] = default

    # Sanitize known numeric fields
    for path, default in numeric_fields.items():
        current = config
        for i, key in enumerate(path[:-1]):
            if key not in current or not isinstance(current[key], dict):
                break
            current = current[key]
        else:
            # We successfully navigated to the parent dict
            final_key = path[-1]
            if final_key in current:
                current[final_key] = sanitize_value(current[final_key], default)

    return config

