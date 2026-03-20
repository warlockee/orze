#!/usr/bin/env python3
"""orze bot: bidirectional messaging companion for orze.

Receives messages from chat platforms, routes them to an LLM for
natural language responses, and sends replies back. Supports:
  - Telegram
  - WeCom (企业微信)
  - DingTalk (钉钉)
  - Feishu (飞书)
  - Generic webhook (any platform)

Built-in commands (/status, /top, /help, /ping) respond instantly.

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
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from typing import List, Optional
from queue import Queue, Empty

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
    "rate_limit": 10,
    "poll_timeout": 30,
    "max_message_len": 2000,
}


def load_config(config_path: str) -> dict:
    """Load orze.yaml and build bot config.

    Looks for config in: bot: (new) → telegram_bot: (legacy) →
    first notification channel as fallback.
    """
    with open(config_path, encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    # New unified config: bot:
    bot_section = cfg.get("bot") or cfg.get("telegram_bot") or {}
    bot_cfg = {**DEFAULTS, **bot_section}
    bot_cfg["_orze"] = cfg

    # Resolve platform-specific credentials from notifications fallback
    platform = bot_cfg.get("type", "telegram")
    if platform == "telegram":
        if not bot_cfg.get("bot_token") or not bot_cfg.get("chat_id"):
            ncfg = cfg.get("notifications") or {}
            for ch in ncfg.get("channels") or []:
                if ch.get("type") == "telegram":
                    bot_cfg.setdefault("bot_token", ch.get("bot_token"))
                    bot_cfg.setdefault("chat_id", ch.get("chat_id"))
                    break

    return bot_cfg


# ---------------------------------------------------------------------------
# Platform-agnostic message
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


def _split_message(text: str, max_len: int = 4096) -> List[str]:
    if len(text) <= max_len:
        return [text]
    chunks = []
    remaining = text
    while remaining:
        if len(remaining) <= max_len:
            chunks.append(remaining)
            break
        cut = remaining[:max_len].rfind("\n\n")
        if cut < max_len // 2:
            cut = remaining[:max_len].rfind("\n")
        if cut < max_len // 4:
            cut = max_len
        chunks.append(remaining[:cut])
        remaining = remaining[cut:].lstrip("\n")
    return chunks


def _post_json(url: str, payload: dict, timeout: int = 10) -> bool:
    """POST JSON to a URL. Returns True on success."""
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        url, data=data, method="POST",
        headers={"Content-Type": "application/json; charset=utf-8",
                 "User-Agent": f"orze-bot/{__version__}"},
    )
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
        return 200 <= resp.status < 300
    except Exception as e:
        logger.debug("POST %s error: %s", url, e)
        return False


# ---------------------------------------------------------------------------
# Platform adapters
# ---------------------------------------------------------------------------

class TelegramAdapter:
    """Telegram Bot API — poll-based."""

    def __init__(self, bot_token: str, chat_id: str,
                 poll_timeout: int = 30, **_kw):
        self.token = bot_token
        self.chat_id = str(chat_id)
        self.poll_timeout = poll_timeout
        self._offset = 0

    def _api(self, method: str, params: dict,
             timeout: int = 10) -> Optional[dict]:
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
            if msg_chat_id != self.chat_id:
                logger.warning("Ignoring message from unauthorized chat: %s",
                               msg_chat_id)
                continue
            user = msg.get("from", {})
            user_name = (user.get("first_name", "") + " " +
                         user.get("last_name", "")).strip()
            messages.append(Message(
                text=text, chat_id=msg_chat_id,
                user=user_name or str(user.get("id", "unknown")),
                update_id=update.get("update_id", 0), raw=update,
            ))
        return messages

    def reply(self, chat_id: str, text: str):
        for chunk in _split_message(text, 4096):
            self._api("sendMessage", {"chat_id": chat_id, "text": chunk})

    def send_typing(self, chat_id: str):
        self._api("sendChatAction",
                  {"chat_id": chat_id, "action": "typing"})


class WeComAdapter:
    """WeCom (企业微信) group bot — webhook out + HTTP listener in."""

    def __init__(self, webhook_url: str, listen_port: int = 8788, **_kw):
        self.webhook_url = webhook_url
        self._queue: Queue = Queue()
        self._start_listener(listen_port)

    def _start_listener(self, port: int):
        queue = self._queue

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else {}
                # WeCom callback format varies; extract text content
                text = (body.get("text", {}).get("content", "")
                        or body.get("content", "")
                        or body.get("text", "")).strip()
                user = body.get("from", {}).get("name", "user")
                if text:
                    queue.put(Message(text=text, chat_id="wecom",
                                     user=user, raw=body))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"errcode":0}')

            def log_message(self, *_args):
                pass  # silence request logs

        server = HTTPServer(("0.0.0.0", port), Handler)
        t = Thread(target=server.serve_forever, daemon=True)
        t.start()
        logger.info("WeCom listener on :%d", port)

    def poll(self) -> List[Message]:
        msgs = []
        try:
            while True:
                msgs.append(self._queue.get_nowait())
        except Empty:
            pass
        if not msgs:
            time.sleep(1)  # avoid busy loop
        return msgs

    def reply(self, chat_id: str, text: str):
        for chunk in _split_message(text, 4096):
            _post_json(self.webhook_url, {
                "msgtype": "text",
                "text": {"content": chunk}
            })

    def send_typing(self, chat_id: str):
        pass  # WeCom doesn't support typing indicators


class DingTalkAdapter:
    """DingTalk (钉钉) custom robot — webhook out + HTTP listener in."""

    def __init__(self, webhook_url: str, listen_port: int = 8788, **_kw):
        self.webhook_url = webhook_url
        self._queue: Queue = Queue()
        self._start_listener(listen_port)

    def _start_listener(self, port: int):
        queue = self._queue

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else {}
                # DingTalk outgoing robot callback format
                text = (body.get("text", {}).get("content", "")
                        or body.get("content", "")).strip()
                user = body.get("senderNick", "user")
                if text:
                    queue.put(Message(text=text, chat_id="dingtalk",
                                     user=user, raw=body))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"errcode":0}')

            def log_message(self, *_args):
                pass

        server = HTTPServer(("0.0.0.0", port), Handler)
        t = Thread(target=server.serve_forever, daemon=True)
        t.start()
        logger.info("DingTalk listener on :%d", port)

    def poll(self) -> List[Message]:
        msgs = []
        try:
            while True:
                msgs.append(self._queue.get_nowait())
        except Empty:
            pass
        if not msgs:
            time.sleep(1)
        return msgs

    def reply(self, chat_id: str, text: str):
        for chunk in _split_message(text, 4096):
            _post_json(self.webhook_url, {
                "msgtype": "text",
                "text": {"content": chunk}
            })

    def send_typing(self, chat_id: str):
        pass


class FeishuAdapter:
    """Feishu (飞书) bot — webhook out + HTTP listener in."""

    def __init__(self, webhook_url: str, listen_port: int = 8788, **_kw):
        self.webhook_url = webhook_url
        self._queue: Queue = Queue()
        self._start_listener(listen_port)

    def _start_listener(self, port: int):
        queue = self._queue

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else {}
                # Feishu event callback — handle challenge for verification
                if "challenge" in body:
                    self.send_response(200)
                    self.end_headers()
                    self.wfile.write(json.dumps(
                        {"challenge": body["challenge"]}).encode())
                    return
                # Extract message text from event
                event = body.get("event", {})
                msg = event.get("message", {})
                content = msg.get("content", "{}")
                try:
                    text = json.loads(content).get("text", "").strip()
                except (json.JSONDecodeError, AttributeError):
                    text = ""
                user = event.get("sender", {}).get("sender_id", {}).get(
                    "open_id", "user")
                if text:
                    queue.put(Message(text=text, chat_id="feishu",
                                     user=user, raw=body))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{}')

            def log_message(self, *_args):
                pass

        server = HTTPServer(("0.0.0.0", port), Handler)
        t = Thread(target=server.serve_forever, daemon=True)
        t.start()
        logger.info("Feishu listener on :%d", port)

    def poll(self) -> List[Message]:
        msgs = []
        try:
            while True:
                msgs.append(self._queue.get_nowait())
        except Empty:
            pass
        if not msgs:
            time.sleep(1)
        return msgs

    def reply(self, chat_id: str, text: str):
        for chunk in _split_message(text, 4096):
            _post_json(self.webhook_url, {
                "msg_type": "text",
                "content": {"text": chunk}
            })

    def send_typing(self, chat_id: str):
        pass


class WebhookAdapter:
    """Generic webhook adapter — works with any platform.

    Config:
        webhook_url: where to POST replies
        listen_port: HTTP port to receive messages (default 8788)
        text_field: JSON path to extract text from incoming POST (default "text")
        user_field: JSON path to extract user name (default "user")
        reply_format: how to wrap reply text in JSON
            default: {"text": "<message>"}
            Or a template string like: {"msgtype": "text", "text": {"content": "{text}"}}
    """

    def __init__(self, webhook_url: str, listen_port: int = 8788,
                 text_field: str = "text", user_field: str = "user",
                 reply_template: dict = None, **_kw):
        self.webhook_url = webhook_url
        self.text_field = text_field
        self.user_field = user_field
        self.reply_template = reply_template
        self._queue: Queue = Queue()
        self._start_listener(listen_port, text_field, user_field)

    def _start_listener(self, port: int, text_field: str, user_field: str):
        queue = self._queue

        def _extract(body, field):
            """Extract nested field like 'text.content' from dict."""
            parts = field.split(".")
            val = body
            for p in parts:
                if isinstance(val, dict):
                    val = val.get(p, "")
                else:
                    return ""
            return str(val).strip() if val else ""

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self):
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length)) if length else {}
                text = _extract(body, text_field)
                user = _extract(body, user_field) or "user"
                if text:
                    queue.put(Message(text=text, chat_id="webhook",
                                     user=user, raw=body))
                self.send_response(200)
                self.end_headers()
                self.wfile.write(b'{"ok":true}')

            def log_message(self, *_args):
                pass

        server = HTTPServer(("0.0.0.0", port), Handler)
        t = Thread(target=server.serve_forever, daemon=True)
        t.start()
        logger.info("Webhook listener on :%d", port)

    def reply(self, chat_id: str, text: str):
        for chunk in _split_message(text, 4096):
            if self.reply_template:
                # Deep-substitute {text} in the template
                payload = json.loads(
                    json.dumps(self.reply_template).replace("{text}", chunk))
            else:
                payload = {"text": chunk}
            _post_json(self.webhook_url, payload)

    def poll(self) -> List[Message]:
        msgs = []
        try:
            while True:
                msgs.append(self._queue.get_nowait())
        except Empty:
            pass
        if not msgs:
            time.sleep(1)
        return msgs

    def send_typing(self, chat_id: str):
        pass


# Adapter registry
_ADAPTERS = {
    "telegram": TelegramAdapter,
    "wecom": WeComAdapter,
    "dingtalk": DingTalkAdapter,
    "feishu": FeishuAdapter,
    "webhook": WebhookAdapter,
}


def create_adapter(cfg: dict):
    """Create the right adapter based on config."""
    platform = cfg.get("type", "telegram")
    cls = _ADAPTERS.get(platform)
    if not cls:
        raise ValueError(f"Unknown bot type: {platform}. "
                         f"Supported: {', '.join(_ADAPTERS)}")
    return cls(**{k: v for k, v in cfg.items()
                  if not k.startswith("_") and k not in (
                      "type", "mode", "model", "allowed_tools",
                      "timeout", "rate_limit", "max_message_len")})


# ---------------------------------------------------------------------------
# Built-in commands (instant, no LLM)
# ---------------------------------------------------------------------------

def _read_status(cfg: dict) -> Optional[dict]:
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
_pending_goal: dict = {}


def cmd_approve(msg: Message, cfg: dict) -> str:
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
    pending = _pending_goal.pop(msg.chat_id, None)
    if not pending:
        return "Nothing pending to reject."
    return "Change discarded."


BUILTIN_COMMANDS = {
    "/ping": cmd_ping,
    "/status": cmd_status,
    "/top": cmd_top,
    "/help": cmd_help,
    "/start": cmd_help,
    "/approve": cmd_approve,
    "/reject": cmd_reject,
    "/yes": cmd_approve,
    "/no": cmd_reject,
}


# ---------------------------------------------------------------------------
# LLM invocation
# ---------------------------------------------------------------------------

def _build_system_prompt(cfg: dict) -> str:
    orze_cfg = cfg["_orze"]
    results_dir = Path(orze_cfg.get("results_dir", "results"))
    ideas_file = orze_cfg.get("ideas_file", "ideas.md")
    goal_file = orze_cfg.get("goal_file", "GOAL.md")

    return f"""You are the orze assistant, responding via chat.

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
- Be concise — this goes to a chat on a phone
- Use plain text, no markdown formatting
- Keep responses under 2000 characters when possible
"""


def invoke_llm(message: str, cfg: dict) -> str:
    mode = cfg.get("mode", "claude")
    if mode == "claude":
        return _invoke_claude(message, cfg)
    elif mode == "script":
        return _invoke_script(message, cfg)
    else:
        return f"Unknown bot mode: {mode}"


def _invoke_claude(message: str, cfg: dict) -> str:
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
        env.pop("CLAUDECODE", None)
        env.pop("CLAUDE_CODE_ENTRYPOINT", None)
        result = subprocess.run(
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=cfg.get("_project_dir", "."), env=env,
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
    script = cfg.get("script")
    if not script:
        return "No script configured for mode: script."
    cmd = [cfg["_orze"].get("python", "python3"), script]
    args = cfg.get("args", [])
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
            cmd, capture_output=True, text=True, timeout=timeout,
            cwd=cfg.get("_project_dir", "."), env=env, input=message,
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
    def __init__(self, max_per_minute: int = 10):
        self.max_per_minute = max_per_minute
        self._timestamps: List[float] = []

    def check(self) -> bool:
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
    import re
    m = re.search(r"```goal\s*\n(.*?)```", response, re.DOTALL)
    if not m:
        return response, None
    before = response[:m.start()].strip()
    return before, m.group(1).strip() + "\n"


def _make_diff(old: str, new: str) -> str:
    import difflib
    old_lines = old.splitlines(keepends=True)
    new_lines = new.splitlines(keepends=True)
    diff = difflib.unified_diff(old_lines, new_lines,
                                fromfile="GOAL.md (current)",
                                tofile="GOAL.md (proposed)",
                                lineterm="")
    return "".join(diff) or "(no visible changes)"


def handle_message(msg: Message, adapter, cfg: dict,
                   rate_limiter: RateLimiter):
    """Process a single incoming message. Adapter is duck-typed."""
    text = msg.text

    max_len = cfg.get("max_message_len", 2000)
    if len(text) > max_len:
        adapter.reply(msg.chat_id, f"Message too long ({len(text)} chars). "
                      f"Max is {max_len}.")
        return

    # Built-in commands
    cmd_key = text.split()[0].lower() if text.startswith("/") else None
    if cmd_key and cmd_key in BUILTIN_COMMANDS:
        response = BUILTIN_COMMANDS[cmd_key](msg, cfg)
        adapter.reply(msg.chat_id, response)
        return

    # Pending confirmation
    if msg.chat_id in _pending_goal:
        lower = text.strip().lower()
        if lower in ("yes", "y", "ok", "confirm", "approve", "lgtm",
                      "好", "确认", "通过"):
            response = cmd_approve(msg, cfg)
            adapter.reply(msg.chat_id, response)
            return
        elif lower in ("no", "n", "reject", "cancel", "discard",
                        "不", "取消", "拒绝"):
            response = cmd_reject(msg, cfg)
            adapter.reply(msg.chat_id, response)
            return
        _pending_goal.pop(msg.chat_id, None)

    # Rate limit
    if not rate_limiter.check():
        adapter.reply(msg.chat_id,
                      "Rate limit exceeded. Try again in a minute.")
        return

    # LLM invocation
    logger.info("Message from %s: %s", msg.user, text[:100])
    adapter.send_typing(msg.chat_id)
    response = invoke_llm(text, cfg)

    # Check for GOAL.md change proposal
    explanation, goal_content = _extract_goal_block(response)
    if goal_content is not None:
        goal_path = Path(cfg["_orze"].get("goal_file", "GOAL.md"))
        current = ""
        if goal_path.exists():
            try:
                current = goal_path.read_text(encoding="utf-8")
            except Exception:
                pass
        diff = _make_diff(current, goal_content)
        _pending_goal[msg.chat_id] = {
            "content": goal_content,
            "diff": diff,
            "timestamp": time.time(),
        }
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
    parser = argparse.ArgumentParser(description="Orze bot")
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

    platform = cfg.get("type", "telegram")

    # Validate platform-specific requirements
    if platform == "telegram":
        if not cfg.get("bot_token") or not cfg.get("chat_id"):
            logger.error(
                "Telegram requires bot_token and chat_id. Configure in "
                "orze.yaml under bot: or telegram_bot:")
            sys.exit(1)
    elif platform in ("wecom", "dingtalk", "feishu", "webhook"):
        if not cfg.get("webhook_url"):
            logger.error(
                "%s requires webhook_url. Configure in orze.yaml under bot:",
                platform)
            sys.exit(1)

    adapter = create_adapter(cfg)
    rate_limiter = RateLimiter(cfg.get("rate_limit", 10))

    logger.info("=" * 50)
    logger.info("Orze bot started (%s)", platform)
    logger.info("Config: %s", config_path)
    logger.info("Mode: %s", cfg.get("mode", "claude"))
    logger.info("=" * 50)

    # Send startup message (only for platforms that support unsolicited messages)
    if platform == "telegram":
        adapter.reply(cfg.get("chat_id", ""),
                      "Orze bot online. Send /help for commands.")
    elif platform in ("wecom", "dingtalk", "feishu"):
        adapter.reply("", "Orze bot online. Send /help for commands.")

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
            break
        except Exception as e:
            logger.error("Poll error: %s", e, exc_info=True)
            time.sleep(5)


if __name__ == "__main__":
    signal.signal(signal.SIGHUP, signal.SIG_IGN)
    main()
