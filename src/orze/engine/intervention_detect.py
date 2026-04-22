"""Pattern-based detection of 'needs human intervention' conditions in role logs."""
import fcntl
import json
import re
import socket
import time
from pathlib import Path
from typing import Optional, Dict, List

PATTERNS: Dict[str, List[str]] = {
    "hf_gated": [
        r"You need to agree to share your contact information",
        r"403.*huggingface",
        r"Access to model .* is restricted",
    ],
    "hf_login": [
        r"huggingface-cli login",
        r"Invalid username or password",
    ],
    "hf_token_missing": [
        r"Token is required",
        r"HF_TOKEN",
    ],
    "gh_login": [
        r"gh auth login",
        r"Bad credentials.*GitHub",
    ],
    "openai_key": [r"OPENAI_API_KEY.*not set"],
    "anthropic_key": [r"ANTHROPIC_API_KEY.*not set"],
    "disk_full": [r"No space left on device"],
    "oom": [r"CUDA out of memory"],
    "sudo_prompt": [r"sudo: a password is required"],
}

_COMPILED = {code: [re.compile(p, re.IGNORECASE) for p in pats] for code, pats in PATTERNS.items()}

COOLDOWN_SEC = 6 * 3600


def detect(log_tail: str, extra_patterns: Optional[dict] = None) -> Optional[tuple]:
    """Return (reason_code, matched_line) of first match, or None.

    extra_patterns: mapping of {reason_code: [regex_str,...]} merged with builtins.
    """
    if not log_tail:
        return None
    patterns = dict(_COMPILED)
    if extra_patterns:
        for code, pats in extra_patterns.items():
            patterns.setdefault(code, [])
            patterns[code] = patterns.get(code, []) + [re.compile(p, re.IGNORECASE) for p in pats]
    for line in log_tail.splitlines():
        for code, pats in patterns.items():
            for pat in pats:
                if pat.search(line):
                    return (code, line.strip())
    return None


def should_notify(state_file: Path, key: str, now: Optional[float] = None) -> bool:
    """Atomic check-and-update of interventions.json. True if caller should fire notify."""
    state_file.parent.mkdir(parents=True, exist_ok=True)
    now = now if now is not None else time.time()
    with open(state_file, "a+") as fp:
        fcntl.flock(fp.fileno(), fcntl.LOCK_EX)
        try:
            fp.seek(0)
            content = fp.read().strip()
            data = json.loads(content) if content else {}
            last = data.get(key, 0)
            if now - last < COOLDOWN_SEC:
                return False
            data[key] = now
            fp.seek(0)
            fp.truncate()
            fp.write(json.dumps(data))
            return True
        finally:
            fcntl.flock(fp.fileno(), fcntl.LOCK_UN)
