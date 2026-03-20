#!/usr/bin/env python3
"""orze bot: bidirectional messaging companion for orze.

Receives messages from Telegram (more platforms later), routes them to
an LLM CLI for natural language responses, and sends replies back.
Built-in commands (/status, /top, /help, /ping) respond instantly
without invoking the LLM.

Runs alongside the orchestrator as a companion process.

Usage:
    python -m orze.agents.bot -c orze.yaml
"""

import argparse
import json
import logging
import os
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import List, Optional

import yaml

from orze import __version__

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [BOT] %(levelname)s %(message)s",
)
logger = logging.getLogger("orze.bot")


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DEFAULTS = {
    "mode": "claude",
    "model": "sonnet",
    "allowed_tools": "Read,Glob,Grep,Bash",
    "timeout": 120,
    "rate_limit": 10,          # max messages per minute
    "poll_timeout": 30,        # Telegram long-poll seconds
    "max_message_len": 2000,   # ignore messages longer than this
}


def load_config(config_path: str) -> dict:
    """Load orze.yaml and build bot config.

    Bot config comes from orze.yaml `telegram_bot:` section, with
    bot_token/chat_id falling back to the first Telegram notification
    channel.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    bot_cfg = {**DEFAULTS, **(cfg.get("telegram_bot") or {})}
    bot_cfg["_orze"] = cfg

    # Resolve bot_token and chat_id
    if not bot_cfg.get("bot_token") or not bot_cfg.get("chat_id"):
        # Fall back to notifications config
        ncfg = cfg.get("notifications") or {}
        for ch in ncfg.get("channels") or []:
            if ch.get("type") == "telegram":
                bot_cfg.setdefault("bot_token", ch.get("bot_token"))
                bot_cfg.setdefault("chat_id", ch.get("chat_id"))
                break

    return bot_cfg


# ---------------------------------------------------------------------------
# Platform adapters
# ---------------------------------------------------------------------------

class Message:
    """Platform-agnostic incoming message."""
    __slots__ = ("text", "chat_id", "user", "update_id", "raw")

    def __init__(self, text: str, chat_id: str, user: str = "",
                 update_id: int = 0, raw: dict = None):
        self.text = text
        self.chat_id = chat_id
        self.user = user
        self.update_id = update_id
        self.raw = raw or {}


class TelegramAdapter:
    """Telegram Bot API adapter using stdlib HTTP (no dependencies)."""

    def __init__(self, bot_token: str, chat_id: str,
                 poll_timeout: int = 30):
        self.token = bot_token
        self.chat_id = str(chat_id)
        self.poll_timeout = poll_timeout
        self._offset = 0

    def _api(self, method: str, params: dict,
             timeout: int = 10) -> Optional[dict]:
        """Call Telegram Bot API. Returns parsed JSON or None."""
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        data = json.dumps(params).encode("utf-8")
        req = urllib.request.Request(
            url, data=data, method="POST",
            headers={"Content-Type": "application/json",
                     "User-Agent": f"orze-bot/{__version__}"},
        )
        try:
            resp = urllib.request.urlopen(req, timeout=timeout)
            result = json.loads(resp.read().decode("utf-8"))
            return result if result.get("ok") else None
        except Exception as e:
            logger.debug("Telegram API %s error: %s", method, e)
            return None

    def poll(self) -> List[Message]:
        """Long-poll for new messages. Returns list of Message objects."""
        params = {"offset": self._offset, "timeout": self.poll_timeout}
        result = self._api("getUpdates", params,
                           timeout=self.poll_timeout + 5)
        if not result:
            return []

        messages = []
        for update in result.get("result", []):
            self._offset = update.get("update_id", 0) + 1
            msg = update.get("message") or {}
            text = (msg.get("text") or "").strip()
            msg_chat_id = str(msg.get("chat", {}).get("id", ""))

            if not text:
                continue

            # Security: only accept messages from configured chat_id
            if msg_chat_id != self.chat_id:
                logger.warning("Ignoring message from unauthorized chat: %s",
                               msg_chat_id)
                continue

            user = msg.get("from", {})
            user_name = (user.get("first_name", "") + " " +
                         user.get("last_name", "")).strip()
            messages.append(Message(
                text=text,
                chat_id=msg_chat_id,
                user=user_name or str(user.get("id", "unknown")),
                update_id=update.get("update_id", 0),
                raw=update,
            ))
        return messages

    def reply(self, chat_id: str, text: str):
        """Send a text reply. Auto-splits messages >4096 chars."""
        for chunk in _split_message(text, 4096):
            self._api("sendMessage", {
                "chat_id": chat_id,
                "text": chunk,
            })

    def send_typing(self, chat_id: str):
        """Send typing indicator."""
        self._api("sendChatAction", {
            "chat_id": chat_id,
            "action": "typing",
        })

    def get_offset(self) -> int:
        return self._offset

    def set_offset(self, offset: int):
        self._offset = offset


def _split_message(text: str, max_len: int = 4096) -> List[str]:
    """Split a message into chunks that fit within max_len."""
    if len(text) <= max_len:
        return [text]

    chunks = []
    remaining = text
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break

        # Try to split on paragraph boundary
        cut = remaining[:max_len].rfind("\n\n")
        if cut < max_len // 2:
            # Try line boundary
            cut = remaining[:max_len].rfind("\n")
        if cut < max_len // 4:
            # Hard cut
            cut = max_len

        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")

    return chunks


# ---------------------------------------------------------------------------
# Built-in commands (instant, no LLM)
# ---------------------------------------------------------------------------

def _read_status(cfg: dict) -> Optional[dict]:
    """Read status.json from results dir."""
    results_dir = Path(cfg["_orze"].get("results_dir", "results"))
    status_file = results_dir / "status.json"
    if not status_file.exists():
        return None
    try:
        return json.loads(status_file.read_text(encoding="utf-8"))
    except Exception:
        return None


def cmd_ping(msg: Message, cfg: dict) -> str:
    return "pong"


def cmd_help(msg: Message, cfg: dict) -> str:
    return (
        "Built-in commands:\n"
        "/ping — check if bot is alive\n"
        "/status — current system status\n"
        "/top — top 5 results\n"
        "/approve — approve pending GOAL.md change\n"
        "/reject — discard pending GOAL.md change\n"
        "/help — this message\n\n"
        "To steer research, just say what you want in plain text.\n"
        "The bot will draft a GOAL.md update and ask for your approval."
    )


def cmd_status(msg: Message, cfg: dict) -> str:
    status = _read_status(cfg)
    if not status:
        return "No status.json found. Is the orchestrator running?"

    active = status.get("active", [])
    active_lines = []
    for a in active[:5]:
        gpu = a.get("gpu", "?")
        idea = a.get("idea_id", "?")
        host = a.get("host", "")
        elapsed = a.get("elapsed_min", 0)
        line = f"  GPU {gpu}: {idea} ({elapsed:.0f}m)"
        if host:
            line += f" [{host}]"
        active_lines.append(line)

    parts = [
        f"Iteration: {status.get('iteration', '?')}",
        f"Host: {status.get('host', '?')}",
        f"Completed: {status.get('completed', 0)} | "
        f"Failed: {status.get('failed', 0)} | "
        f"Queued: {status.get('queue_depth', 0)}",
        f"Free GPUs: {status.get('free_gpus', [])}",
        f"Disk: {status.get('disk_free_gb', '?')} GB free",
    ]

    if active_lines:
        parts.append(f"\nActive ({len(active)}):")
        parts.extend(active_lines)
    else:
        parts.append("\nNo active training.")

    return "\n".join(parts)


def cmd_top(msg: Message, cfg: dict) -> str:
    status = _read_status(cfg)
    if not status:
        return "No status.json found. Is the orchestrator running?"

    top = status.get("top_results", [])
    if not top:
        return "No completed experiments yet."

    primary = cfg["_orze"].get("report", {}).get("primary_metric",
                                                  "test_accuracy")
    lines = ["Top 5 results:"]
    for i, r in enumerate(top[:5], 1):
        idea_id = r.get("idea_id", "?")
        val = r.get(primary, r.get("value", "?"))
        if isinstance(val, float):
            val = f"{val:.4f}"
        lines.append(f"  {i}. {idea_id} — {primary}: {val}")
    return "\n".join(lines)


# Pending GOAL.md changes awaiting user confirmation
# Key: chat_id, Value: {"content": str, "diff": str, "timestamp": float}
_pending_goal: dict = {}


def cmd_approve(msg: Message, cfg: dict) -> str:
    """Approve a pending GOAL.md change."""
    pending = _pending_goal.pop(msg.chat_id, None)
    if not pending:
        return "Nothing pending to approve."
    goal_path = Path(cfg["_orze"].get("goal_file", "GOAL.md"))
    try:
        goal_path.write_text(pending["content"], encoding="utf-8")
        return "GOAL.md updated. Research agent will pick this up in ~30s."
    except Exception as e:
        return f"Failed to write GOAL.md: {e}"


def cmd_reject(msg: Message, cfg: dict) -> str:
    """Reject a pending GOAL.md change."""
    pending = _pending_goal.pop(msg.chat_id, None)
    if not pending:
        return "Nothing pending to reject."
    return "Change discarded."


BUILTIN_COMMANDS = {
    "/ping": cmd_ping,
    "/status": cmd_status,
    "/top": cmd_top,
    "/help": cmd_help,
    "/start": cmd_help,  # Telegram sends /start on first interaction
    "/approve": cmd_approve,
    "/reject": cmd_reject,
    "/yes": cmd_approve,
    "/no": cmd_reject,
}


# ---------------------------------------------------------------------------
# LLM invocation
# ---------------------------------------------------------------------------

def _build_system_prompt(cfg: dict) -> str:
    """Build the system prompt that gives the LLM context about orze."""
    orze_cfg = cfg["_orze"]
    results_dir = Path(orze_cfg.get("results_dir", "results"))
    ideas_file = orze_cfg.get("ideas_file", "ideas.md")

    goal_file = orze_cfg.get("goal_file", "GOAL.md")

    return f"""You are the orze assistant, responding via Telegram chat.

## Available files
- Goal: {goal_file} (research direction — the user's steering wheel)
- Status: {results_dir / 'status.json'} (iteration, active jobs, GPUs, queue, disk, top results)
- Report: {results_dir / 'report.md'} (full leaderboard)
- Ideas: {ideas_file} (experiment definitions)
- Results: {results_dir}/<idea-id>/metrics.json (per-experiment metrics)

## What you can do
- Read any file to answer questions about experiments, results, status
- Analyze results and provide insights

## Steering research direction
When the user wants to change research direction (e.g., "try lower learning rates",
"stop exploring transformers", "focus on augmentation"), you MUST:
1. Read the current {goal_file}
2. Draft the updated version
3. Output it inside a ```goal``` code block (the bot will handle the rest)

Example response:
  "Here's my proposed update to GOAL.md:"
  ```goal
  # Research Goal
  ... updated content ...
  ```

The bot will show the diff to the user for approval before writing.
Do NOT use Write or Edit tools on {goal_file} — always use the ```goal``` block.

## What you must NOT do
- Write or edit {goal_file} directly (use ```goal``` block instead)
- Modify orze source code, bot.py, or orze.yaml
- Delete any files
- Push to git
- Run training or eval scripts

## Response format
- Be concise — this goes to a Telegram chat on a phone
- Use plain text, no markdown formatting (Telegram doesn't render it well)
- Keep responses under 2000 characters when possible
"""


def invoke_llm(message: str, cfg: dict) -> str:
    """Invoke an LLM to handle a natural language message.

    Supports mode: claude (Claude CLI) and mode: script (any script).
    """
    mode = cfg.get("mode", "claude")

    if mode == "claude":
        return _invoke_claude(message, cfg)
    elif mode == "script":
        return _invoke_script(message, cfg)
    else:
        return f"Unknown bot mode: {mode}"


def _invoke_claude(message: str, cfg: dict) -> str:
    """Invoke Claude CLI with the user's message."""
    system_prompt = _build_system_prompt(cfg)
    prompt = f"{system_prompt}\n\n## User message\n{message}"

    cmd = [
        "claude", "-p", prompt,
        "--output-format", "text",
        "--dangerously-skip-permissions",
    ]

    model = cfg.get("model", "sonnet")
    if model:
        cmd.extend(["--model", model])

    allowed_tools = cfg.get("allowed_tools", "Read,Glob,Grep,Bash")
    if allowed_tools:
        cmd.extend(["--allowedTools", allowed_tools])

    timeout = cfg.get("timeout", 120)

    try:
        env = os.environ.copy()
        # Avoid nesting issues if running inside Claude Code
        env.pop("CLAUDECODE", None)
        env.pop("CLAUDE_CODE_ENTRYPOINT", None)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cfg.get("_project_dir", "."),
            env=env,
        )
        response = (result.stdout or "").strip()
        if not response and result.stderr:
            logger.warning("Claude stderr: %s", result.stderr[:500])
            return "LLM returned no response. Check bot logs."
        return response or "No response from LLM."
    except subprocess.TimeoutExpired:
        return f"Request timed out after {timeout}s. Try a simpler question."
    except FileNotFoundError:
        return "Claude CLI not found. Install it or switch to mode: script."
    except Exception as e:
        logger.error("LLM invocation failed: %s", e)
        return "Internal error invoking LLM. Check bot logs."


def _invoke_script(message: str, cfg: dict) -> str:
    """Invoke a custom script with the user's message."""
    script = cfg.get("script")
    if not script:
        return "No script configured for mode: script."

    cmd = [cfg["_orze"].get("python", "python3"), script]

    args = cfg.get("args", [])
    # Template substitution
    orze_cfg = cfg["_orze"]
    template_vars = {
        "results_dir": orze_cfg.get("results_dir", "results"),
        "ideas_file": orze_cfg.get("ideas_file", "ideas.md"),
        "message": message,
    }
    for a in args:
        try:
            cmd.append(str(a).format(**template_vars))
        except (KeyError, IndexError, ValueError):
            cmd.append(str(a))

    timeout = cfg.get("timeout", 120)

    try:
        env = os.environ.copy()
        for k, v in (cfg.get("env") or {}).items():
            env[k] = str(v)

        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=cfg.get("_project_dir", "."),
            env=env,
            input=message,
        )
        return (result.stdout or "").strip() or "Script returned no output."
    except subprocess.TimeoutExpired:
        return f"Script timed out after {timeout}s."
    except Exception as e:
        logger.error("Script invocation failed: %s", e)
        return "Internal error running script. Check bot logs."


# ---------------------------------------------------------------------------
# Rate limiter
# ---------------------------------------------------------------------------

class RateLimiter:
    """Simple sliding-window rate limiter."""

    def __init__(self, max_per_minute: int = 10):
        self.max_per_minute = max_per_minute
        self._timestamps: List[float] = []

    def check(self) -> bool:
        """Returns True if the request is allowed."""
        now = time.time()
        self._timestamps = [t for t in self._timestamps if now - t < 60]
        if len(self._timestamps) >= self.max_per_minute:
            return False
        self._timestamps.append(now)
        return True


# ---------------------------------------------------------------------------
# Message handler
# ---------------------------------------------------------------------------

def _extract_goal_block(response: str):
    """Extract content from ```goal ... ``` block in LLM response.

    Returns (text_before_block, goal_content) or (response, None).
    """
    import re
    m = re.search(r"```goal\s*\n(.*?)```", response, re.DOTALL)
    if not m:
        return response, None
    # Text before the code block (the LLM's explanation)
    before = response[:m.start()].strip()
    return before, m.group(1).strip() + "\n"


def _make_diff(old: str, new: str) -> str:
    """Create a simple line-level diff for Telegram display."""
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    import difflib
    diff = difflib.unified_diff(old_lines, new_lines,
                                fromfile="GOAL.md (current)",
                                tofile="GOAL.md (proposed)",
                                lineterm="")
    return "".join(diff) or "(no visible changes)"


def handle_message(msg: Message, adapter: TelegramAdapter,
                   cfg: dict, rate_limiter: RateLimiter):
    """Process a single incoming message."""
    text = msg.text

    # Ignore overly long messages
    max_len = cfg.get("max_message_len", 2000)
    if len(text) > max_len:
        adapter.reply(msg.chat_id, f"Message too long ({len(text)} chars). "
                      f"Max is {max_len}.")
        return

    # Check built-in commands
    cmd_key = text.split()[0].lower() if text.startswith("/") else None
    if cmd_key and cmd_key in BUILTIN_COMMANDS:
        response = BUILTIN_COMMANDS[cmd_key](msg, cfg)
        adapter.reply(msg.chat_id, response)
        return

    # Handle pending confirmation: bare "yes"/"no"/"ok" without slash
    if msg.chat_id in _pending_goal:
        lower = text.strip().lower()
        if lower in ("yes", "y", "ok", "confirm", "approve", "lgtm"):
            response = cmd_approve(msg, cfg)
            adapter.reply(msg.chat_id, response)
            return
        elif lower in ("no", "n", "reject", "cancel", "discard"):
            response = cmd_reject(msg, cfg)
            adapter.reply(msg.chat_id, response)
            return
        # Any other text: discard pending and process as new message
        _pending_goal.pop(msg.chat_id, None)

    # Rate limit for LLM calls
    if not rate_limiter.check():
        adapter.reply(msg.chat_id,
                      "Rate limit exceeded. Try again in a minute.")
        return

    # Send typing indicator and invoke LLM
    logger.info("Message from %s: %s", msg.user, text[:100])
    adapter.send_typing(msg.chat_id)
    response = invoke_llm(text, cfg)

    # Check if LLM proposed a GOAL.md change
    explanation, goal_content = _extract_goal_block(response)
    if goal_content is not None:
        # Read current GOAL.md for diff
        goal_path = Path(cfg["_orze"].get("goal_file", "GOAL.md"))
        current = ""
        if goal_path.exists():
            try:
                current = goal_path.read_text(encoding="utf-8")
            except Exception:
                pass

        diff = _make_diff(current, goal_content)

        # Store pending change
        _pending_goal[msg.chat_id] = {
            "content": goal_content,
            "diff": diff,
            "timestamp": time.time(),
        }

        # Send explanation + diff + prompt for confirmation
        if explanation:
            adapter.reply(msg.chat_id, explanation)
        adapter.reply(msg.chat_id, f"Proposed diff:\n\n{diff}")
        adapter.reply(msg.chat_id, "Reply yes to apply, no to discard.")
    else:
        adapter.reply(msg.chat_id, response)
    logger.info("Replied (%d chars)", len(response))


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Orze Telegram bot")
    parser.add_argument("-c", "--config", default="orze.yaml",
                        help="Path to orze.yaml")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Debug logging")
    args = parser.parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    config_path = Path(args.config).resolve()
    project_dir = config_path.parent

    cfg = load_config(str(config_path))
    cfg["_project_dir"] = str(project_dir)

    # Validate required config
    bot_token = cfg.get("bot_token")
    chat_id = cfg.get("chat_id")
    if not bot_token or not chat_id:
        logger.error(
            "No bot_token/chat_id found. Configure telegram_bot: section "
            "in orze.yaml or add a Telegram notification channel.")
        sys.exit(1)

    adapter = TelegramAdapter(
        bot_token=bot_token,
        chat_id=str(chat_id),
        poll_timeout=cfg.get("poll_timeout", 30),
    )

    rate_limiter = RateLimiter(cfg.get("rate_limit", 10))

    logger.info("=" * 50)
    logger.info("Orze bot started")
    logger.info("Config: %s", config_path)
    logger.info("Mode: %s", cfg.get("mode", "claude"))
    logger.info("Chat ID: %s", chat_id)
    logger.info("=" * 50)

    # Send startup message
    adapter.reply(str(chat_id), "Orze bot online. Send /help for commands.")

    while True:
        try:
            messages = adapter.poll()
            for msg in messages:
                try:
                    handle_message(msg, adapter, cfg, rate_limiter)
                except Exception as e:
                    logger.error("Error handling message: %s", e,
                                 exc_info=True)
                    try:
                        adapter.reply(msg.chat_id,
                                      "Internal error. Check bot logs.")
                    except Exception:
                        pass

        except KeyboardInterrupt:
            logger.info("Shutting down...")
            adapter.reply(str(chat_id), "Orze bot going offline.")
            break
        except Exception as e:
            logger.error("Poll error: %s", e, exc_info=True)
            time.sleep(5)  # Back off on errors


if __name__ == "__main__":
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    main()
