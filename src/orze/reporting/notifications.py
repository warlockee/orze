import datetime
import html as html_mod
import json
import logging
import socket
import urllib.request
from typing import Dict, Optional

from orze import __version__
from orze.reporting.leaderboard import _format_report_text

logger = logging.getLogger("orze")


def _notify_send(url: str, payload: dict,
                 headers: Optional[Dict[str, str]] = None,
                 timeout: int = 10):
    """POST JSON payload to a URL. Never raises."""
    try:
        data = json.dumps(payload).encode("utf-8")
        req_headers = {
            "Content-Type": "application/json",
            "User-Agent": f"orze/{__version__}",
        }
        if headers:
            req_headers.update(headers)
        req = urllib.request.Request(url, data=data, method="POST",
                                     headers=req_headers)
        urllib.request.urlopen(req, timeout=timeout)
    except Exception as e:
        logger.warning("Notification failed (%s): %s", url[:60], e)


def _format_leaderboard(data: dict, bold_fn=str, escape_fn=str) -> str:
    """Format top 10 leaderboard lines from data['leaderboard'].
    escape_fn is applied to all text content (needed for Telegram HTML)."""
    board = data.get("leaderboard", [])
    if not board:
        return ""
    metric = escape_fn(str(data.get("metric_name", "score")))
    lines = [f"\nTop {len(board)} ({metric}):"]
    for i, entry in enumerate(board, 1):
        val = entry.get("value")
        val_str = escape_fn(f"{val:.4f}" if isinstance(val, float) else str(val))
        marker = escape_fn(" <-" if entry["id"] == data.get("idea_id") else "")
        title = escape_fn(str(entry.get("title", ""))[:30])
        line = f"#{i} {escape_fn(str(entry['id']))}: {val_str} {title}{marker}"
        if i == 1:
            line = bold_fn(line)
        lines.append(line)

    # Append view leaderboards (e.g. edge)
    for vname, vdata in (data.get("view_leaderboards") or {}).items():
        vtitle = escape_fn(str(vdata.get("title", vname)))
        entries = vdata.get("entries") or []
        if not entries:
            continue
        lines.append(f"\n{bold_fn(vtitle)} (top {len(entries)}):")
        for i, entry in enumerate(entries, 1):
            val = entry.get("value")
            val_str = escape_fn(f"{val:.4f}" if isinstance(val, float) else str(val))
            marker = escape_fn(" <-" if entry["id"] == data.get("idea_id") else "")
            title = escape_fn(str(entry.get("title", ""))[:30])
            line = f"#{i} {escape_fn(str(entry['id']))}: {val_str} {title}{marker}"
            if i == 1:
                line = bold_fn(line)
            lines.append(line)

    return "\n".join(lines)


def _format_slack(event: str, data: dict) -> dict:
    """Format notification for Slack webhook."""
    if event == "report":
        return {"text": f"```\n{_format_report_text(data)}\n```"}
    if event in ("started", "shutdown"):
        icon = ":arrow_forward:" if event == "started" else ":stop_button:"
        host = data.get("host", socket.gethostname())
        return {"text": f"{icon} *Orze {event}* on `{host}`\n{data.get('message', '')}"}
    if event == "heartbeat":
        host = data.get("host", socket.gethostname())
        return {"text": (f":green_heart: *Heartbeat* on `{host}` | "
                         f"Iter {data.get('iteration', '?')} | "
                         f"Up {data.get('uptime', '?')} | "
                         f"{data.get('training', 0)}T/{data.get('eval', 0)}E/"
                         f"{data.get('free', 0)}F GPUs | "
                         f"Done {data.get('completed', 0)} | "
                         f"Q {data.get('queued', 0)}")}
    if event == "milestone":
        return {"text": f":dart: *Milestone: {data.get('count', '?')} experiments completed!*"}
    if event == "disk_warning":
        return {"text": f":warning: *Low disk* on `{data.get('host', socket.gethostname())}` — "
                        f"only {data.get('free_gb', '?')}GB free"}
    if event == "stall":
        return {"text": f":rotating_light: *{data.get('reason', 'Stalled')}*: "
                        f"`{data.get('idea_id', '?')}` on GPU {data.get('gpu', '?')}"}
    if event == "role_summary":
        bd = data.get("breakdown") or str(data.get("new_ideas", 0))
        return {"text": f":test_tube: *{data.get('role', '?')}* finished | "
                        f"{bd} new ideas | {data.get('queued', '?')} queued"}
    if event == "upgrading":
        host = data.get("host", socket.gethostname())
        return {"text": f":arrows_counterclockwise: *Orze upgrading* on `{host}`: "
                        f"v{data.get('from_version', '?')} -> v{data.get('to_version', '?')}"}
    if event == "watchdog_restart":
        host = data.get("host", socket.gethostname())
        reason = data.get("reason", "unknown")
        return {"text": f":dog: *Watchdog restarted orze* on `{host}` (reason: {reason})"}
    if event == "new_best":
        prev = data.get("prev_best_id", "none")
        text = (f":trophy: *NEW BEST* `{data['idea_id']}`: {data['title']}\n"
                f"{data['metric_name']}: *{data['metric_value']}*"
                f" (was `{prev}`)")
    elif event == "failed":
        err = str(data.get("error") or "unknown")[:200]
        text = (f":x: *FAILED* `{data['idea_id']}`: {data['title']}\n"
                f"Error: {err}")
    else:
        t_str = ""
        try:
            t = data.get("training_time")
            if t is not None:
                t_str = f" in {float(t):.0f}s"
        except (ValueError, TypeError):
            pass
        text = (f":white_check_mark: *Completed* `{data.get('idea_id', '?')}`: "
                f"{data.get('title', '?')}\n"
                f"{data.get('metric_name', '?')}: {data.get('metric_value', '?')}"
                f" (rank #{data.get('rank', '?')})"
                f"{t_str}")
    text += _format_leaderboard(data, lambda s: f"*{s}*")
    return {"text": text}


def _format_discord(event: str, data: dict) -> dict:
    """Format notification for Discord webhook."""
    if event == "report":
        return {"content": f"```\n{_format_report_text(data)}\n```"}
    if event in ("started", "shutdown"):
        icon = "\u25b6\ufe0f" if event == "started" else "\u23f9\ufe0f"
        host = data.get("host", socket.gethostname())
        return {"content": f"{icon} **Orze {event}** on `{host}`\n{data.get('message', '')}"}
    if event == "heartbeat":
        host = data.get("host", socket.gethostname())
        return {"content": (f"\U0001f49a **Heartbeat** on `{host}` | "
                            f"Iter {data.get('iteration', '?')} | "
                            f"Up {data.get('uptime', '?')} | "
                            f"{data.get('training', 0)}T/{data.get('eval', 0)}E/"
                            f"{data.get('free', 0)}F GPUs | "
                            f"Done {data.get('completed', 0)} | Q {data.get('queued', 0)}")}
    if event == "milestone":
        return {"content": f"\U0001f3af **Milestone: {data.get('count', '?')} experiments completed!**"}
    if event == "disk_warning":
        return {"content": f"\u26a0\ufe0f **Low disk** on `{data.get('host', socket.gethostname())}` — "
                           f"only {data.get('free_gb', '?')}GB free"}
    if event == "stall":
        return {"content": f"\U0001f6a8 **{data.get('reason', 'Stalled')}**: "
                           f"`{data.get('idea_id', '?')}` on GPU {data.get('gpu', '?')}"}
    if event == "role_summary":
        bd = data.get("breakdown") or str(data.get("new_ideas", 0))
        return {"content": f"\U0001f9ea **{data.get('role', '?')}** finished | "
                           f"{bd} new ideas | {data.get('queued', '?')} queued"}
    if event == "upgrading":
        host = data.get("host", socket.gethostname())
        return {"content": f"\U0001f504 **Orze upgrading** on `{host}`: "
                           f"v{data.get('from_version', '?')} -> v{data.get('to_version', '?')}"}
    if event == "watchdog_restart":
        host = data.get("host", socket.gethostname())
        reason = data.get("reason", "unknown")
        return {"content": f"\U0001f415 **Watchdog restarted orze** on `{host}` (reason: {reason})"}
    if event == "new_best":
        prev = data.get("prev_best_id", "none")
        content = (f"**NEW BEST** `{data['idea_id']}`: {data['title']}\n"
                   f"{data['metric_name']}: **{data['metric_value']}**"
                   f" (was `{prev}`)")
    elif event == "failed":
        err = str(data.get("error") or "unknown")[:200]
        content = (f"**FAILED** `{data['idea_id']}`: {data['title']}\n"
                   f"Error: {err}")
    else:
        t_str = ""
        try:
            t = data.get("training_time")
            if t is not None:
                t_str = f" in {float(t):.0f}s"
        except (ValueError, TypeError):
            pass
        content = (f"**Completed** `{data.get('idea_id', '?')}`: {data.get('title', '?')}\n"
                   f"{data.get('metric_name', '?')}: {data.get('metric_value', '?')}"
                   f" (rank #{data.get('rank', '?')})"
                   f"{t_str}")
    content += _format_leaderboard(data, lambda s: f"**{s}**")
    return {"content": content}


def _format_telegram(event: str, data: dict, channel_cfg: dict) -> tuple:
    """Format notification for Telegram Bot API using HTML parse_mode.
    Returns (url, payload). Uses HTML to avoid MarkdownV2 escaping issues."""
    esc = html_mod.escape
    token = channel_cfg["bot_token"]
    chat_id = channel_cfg["chat_id"]
    url = f"https://api.telegram.org/bot{token}/sendMessage"

    if event == "report":
        text = f"<pre>\n{esc(_format_report_text(data))}\n</pre>"
        return url, {"chat_id": chat_id, "text": text,
                     "parse_mode": "HTML"}

    if event in ("started", "shutdown"):
        host = esc(data.get("host", socket.gethostname()))
        msg = esc(str(data.get("message", "")))
        icon = "\u25b6\ufe0f" if event == "started" else "\u23f9\ufe0f"
        text = f"{icon} <b>Orze {event}</b> on <code>{host}</code>\n{msg}"
        return url, {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}

    if event == "heartbeat":
        host = esc(data.get("host", socket.gethostname()))
        lines = [f"\U0001f49a <b>Heartbeat</b> on <code>{host}</code>"]
        lines.append(f"Iter {data.get('iteration', '?')} | "
                      f"Up {esc(str(data.get('uptime', '?')))} | "
                      f"GPUs {data.get('training', 0)}T/{data.get('eval', 0)}E/"
                      f"{data.get('free', 0)}F")
        lines.append(f"Completed {data.get('completed', 0)} | "
                      f"Queued {data.get('queued', 0)} | "
                      f"Failed {data.get('failed', 0)}")
        if data.get("eval_backlog"):
            lines.append(f"Eval backlog: {data['eval_backlog']}")
        if data.get("rate"):
            lines.append(f"Rate: {data['rate']}")
        return url, {"chat_id": chat_id, "text": "\n".join(lines),
                     "parse_mode": "HTML"}

    if event == "milestone":
        text = (f"\U0001f3af <b>Milestone: {esc(str(data.get('count', '?')))} "
                f"experiments completed!</b>")
        return url, {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}

    if event == "disk_warning":
        text = (f"\u26a0\ufe0f <b>Low disk</b> on <code>"
                f"{esc(data.get('host', socket.gethostname()))}</code>\n"
                f"Only {data.get('free_gb', '?')}GB free")
        return url, {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}

    if event == "stall":
        reason = esc(str(data.get("reason", "stalled")))
        text = (f"\U0001f6a8 <b>{reason}</b>: <code>"
                f"{esc(str(data.get('idea_id', '?')))}</code> on GPU "
                f"{data.get('gpu', '?')}")
        return url, {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}

    if event == "role_summary":
        role = esc(str(data.get("role", "?")))
        bd = esc(str(data.get("breakdown") or data.get("new_ideas", 0)))
        text = (f"\U0001f9ea <b>{role}</b> finished | "
                f"{bd} new ideas | "
                f"{data.get('queued', '?')} queued")
        return url, {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}

    if event == "upgrading":
        host = esc(data.get("host", socket.gethostname()))
        frm = esc(str(data.get("from_version", "?")))
        to = esc(str(data.get("to_version", "?")))
        text = (f"\U0001f504 <b>Orze upgrading</b> on <code>{host}</code>\n"
                f"v{frm} -> v{to}")
        return url, {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}

    if event == "watchdog_restart":
        host = esc(data.get("host", socket.gethostname()))
        reason = esc(str(data.get("reason", "unknown")))
        text = (f"\U0001f415 <b>Watchdog restarted orze</b> on "
                f"<code>{host}</code> (reason: {reason})")
        return url, {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}

    idea_id = esc(str(data.get("idea_id", "")))
    title = esc(str(data.get("title", "")))

    if event == "new_best":
        prev = esc(str(data.get("prev_best_id", "none")))
        metric = esc(str(data.get("metric_name", "")))
        val = esc(str(data.get("metric_value", "")))
        text = (f"<b>NEW BEST</b> <code>{idea_id}</code>: {title}\n"
                f"{metric}: <b>{val}</b>"
                f" (was <code>{prev}</code>)")
    elif event == "failed":
        err = esc(str(data.get("error") or "unknown")[:200])
        text = (f"<b>FAILED</b> <code>{idea_id}</code>: {title}\n"
                f"Error: {err}")
    else:
        t_str = ""
        try:
            t = data.get("training_time")
            if t is not None:
                t_str = f" in {float(t):.0f}s"
        except (ValueError, TypeError):
            pass
        metric = esc(str(data.get("metric_name", "")))
        val = esc(str(data.get("metric_value", "")))
        rank = esc(str(data.get("rank", "?")))
        text = (f"<b>Completed</b> <code>{idea_id}</code>: {title}\n"
                f"{metric}: {val}"
                f" (rank #{rank})"
                f"{t_str}")
    text += _format_leaderboard(data, lambda s: f"<b>{s}</b>", escape_fn=esc)

    return url, {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}


def notify(event: str, data: dict, cfg: dict):
    """Send notifications for an event to all configured channels. Never raises."""
    try:
        ncfg = cfg.get("notifications") or {}
        if not ncfg.get("enabled", False):
            return

        logger.info("Sending notification for event: %s", event)
        global_on = ncfg.get("on") or ["completed", "failed", "new_best"]
        # Lifecycle/system events always delivered (not filtered)
        lifecycle = {"started", "shutdown", "heartbeat", "milestone",
                     "disk_warning", "stall", "role_summary", "upgrading",
                     "watchdog_restart"}

        for ch in (ncfg.get("channels") or []):
            ch_on = ch.get("on") or global_on
            if event not in lifecycle and event not in ch_on:
                continue

            ch_type = ch.get("type", "webhook")
            if ch_type == "slack":
                _notify_send(ch["webhook_url"], _format_slack(event, data))
            elif ch_type == "discord":
                _notify_send(ch["webhook_url"], _format_discord(event, data))
            elif ch_type == "telegram":
                url, payload = _format_telegram(event, data, ch)
                _notify_send(url, payload)
            elif ch_type == "webhook":
                payload = {"event": event, "data": data,
                           "host": socket.gethostname(),
                           "timestamp": datetime.datetime.now().isoformat()}
                _notify_send(ch["url"], payload, headers=ch.get("headers"))
            else:
                logger.warning("Unknown notification channel: %s", ch_type)
    except Exception as e:
        logger.warning("Notification dispatch error: %s", e)
