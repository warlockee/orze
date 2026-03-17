"""Skill loader — composable prompt fragments for Orze roles.

CALLING SPEC:
    compose_skills(role_cfg, project_root, template_vars=None) -> str
        role_cfg:       dict with 'skills' list or 'rules_file' (legacy)
        project_root:   Path to project directory
        template_vars:  dict of {key: value} for substitution, or None to skip
        returns:        composed prompt string

    load_builtin(name) -> Skill
        name: one of 'core', 'research', 'ops', 'setup'
        returns: Skill namedtuple(name, content)
        raises: FileNotFoundError if name is unknown

LEGACY PATH:
    If role_cfg has 'rules_file' instead of 'skills', reads the file and
    optionally substitutes template vars. Identical behavior to old code.

SKILLS PATH:
    If role_cfg has 'skills' list, loads each ref in order:
      - '@core', '@research', etc -> built-in .skill.md from this package
      - './path.md' or 'path.md' -> project file relative to project_root
    Concatenates with separator. Substitutes template vars if provided.
"""

import logging
from collections import namedtuple
from pathlib import Path
from typing import Dict, List, Optional

import yaml

logger = logging.getLogger("orze")

Skill = namedtuple("Skill", ["name", "content"])

_SKILLS_DIR = Path(__file__).parent

# Valid built-in skill names
_BUILTINS = {"core", "research", "ops", "setup"}


def parse_frontmatter(text: str) -> tuple:
    """Split ---yaml--- header from content.

    Returns (meta_dict, body_str). If no frontmatter, returns ({}, text).
    """
    if not text.startswith("---"):
        return {}, text
    parts = text.split("---", 2)
    if len(parts) < 3:
        return {}, text
    try:
        meta = yaml.safe_load(parts[1]) or {}
    except yaml.YAMLError:
        meta = {}
    return meta, parts[2].strip()


def load_builtin(name: str) -> Skill:
    """Load a built-in skill from this package's .skill.md files."""
    if name not in _BUILTINS:
        raise FileNotFoundError(
            f"Unknown built-in skill '@{name}'. "
            f"Available: {', '.join(sorted('@' + b for b in _BUILTINS))}")
    path = _SKILLS_DIR / f"{name}.skill.md"
    if not path.exists():
        raise FileNotFoundError(f"Built-in skill file not found: {path}")
    text = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(text)
    return Skill(name=meta.get("name", name), content=body)


def load_file(path: Path) -> Skill:
    """Load a skill from a project file.

    Plain .md without frontmatter gets name from filename stem.
    """
    if not path.exists():
        raise FileNotFoundError(f"Skill file not found: {path}")
    text = path.read_text(encoding="utf-8")
    meta, body = parse_frontmatter(text)
    name = meta.get("name", path.stem)
    return Skill(name=name, content=body)


def _substitute(text: str, template_vars: Dict[str, str]) -> str:
    """Safe template var substitution using str.replace."""
    for k, v in template_vars.items():
        text = text.replace(f"{{{k}}}", str(v))
    return text


def compose_skills(role_cfg: dict, project_root: Path,
                   template_vars: Optional[Dict[str, str]] = None) -> str:
    """Compose prompt from skills list or legacy rules_file.

    Returns the composed prompt string ready to pass to an LLM.
    """
    # Legacy path: rules_file
    if "skills" not in role_cfg:
        rules_file = role_cfg.get("rules_file")
        if not rules_file:
            return ""
        path = Path(rules_file)
        if not path.is_absolute():
            path = project_root / path
        if not path.exists():
            logger.warning("Rules file not found: %s", path)
            return ""
        content = path.read_text(encoding="utf-8")
        if template_vars:
            content = _substitute(content, template_vars)
        return content

    # Skills path: compose from list
    skills_list = role_cfg["skills"]
    if not isinstance(skills_list, list):
        logger.warning("'skills' must be a list, got %s", type(skills_list).__name__)
        return ""

    loaded: List[Skill] = []
    for ref in skills_list:
        ref = str(ref).strip()
        if ref.startswith("@"):
            name = ref[1:]
            try:
                loaded.append(load_builtin(name))
            except FileNotFoundError as e:
                logger.warning("Skill %s: %s", ref, e)
        else:
            path = Path(ref)
            if not path.is_absolute():
                path = project_root / path
            try:
                loaded.append(load_file(path))
            except FileNotFoundError as e:
                logger.warning("Skill %s: %s", ref, e)

    if not loaded:
        return ""

    # Compose
    sections = [skill.content for skill in loaded]
    composed = "\n\n---\n\n".join(sections)

    if template_vars:
        composed = _substitute(composed, template_vars)

    logger.info("Composed %d skills: %s",
                len(loaded), ", ".join(s.name for s in loaded))
    return composed
