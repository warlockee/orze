"""Notification delivery to Slack, Discord, Telegram, and generic webhooks.

CALLING SPEC:
    notify(event: str, data: dict, cfg: dict) -> None
        Send notifications for an event to all configured channels. Never
        raises. event is one of: completed, failed, new_best, report,
        started, shutdown, heartbeat, milestone, disk_warning, stall,
        role_summary, upgrading, watchdog_restart, plateau, needs_intervention.
        data is event-specific (e.g. idea_id, title, metric_value, rank,
        leaderboard). cfg must contain 'notifications' key with 'enabled',
        'on' (event filter list), and 'channels' (list of channel configs
        with type: slack|discord|telegram|webhook).
"""
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
        logger.error("Notification delivery failed (%s): %s: %s",
                     url[:60], type(e).__name__, e)


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
        if isinstance(val, float):
            val_str = escape_fn(f"{val:.4f}")
        elif isinstance(val, (int,)):
            val_str = escape_fn(str(val))
        elif val is not None:
            val_str = escape_fn(str(val))
        else:
            val_str = "—"
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
        completed = data.get("completed", 0)
        failed = data.get("failed", 0)
        queued = data.get("queued", 0)
        running = data.get("running", data.get("training", 0) + data.get("eval", 0))
        total_gpus = data.get("total_gpus", running + data.get("free", 0))
        active_gpus = total_gpus - data.get("free", 0)
        lines = [
            f":bar_chart: *Orze Status* — `{host}`",
            (f":white_check_mark: {completed} completed | :x: {failed} failed | "
             f":hourglass_flowing_sand: {queued} queued | :arrows_counterclockwise: {running} running"),
        ]
        best_val = data.get("best_val")
        if best_val is not None:
            metric = data.get("best_metric", "score")
            best_id = data.get("best_id", "?")
            val_str = f"{best_val:.4f}" if isinstance(best_val, float) else str(best_val)
            lines.append(f":trophy: Best: {val_str} {metric} (`{best_id}`)")
        lines.append(f":computer: GPUs: {active_gpus}/{total_gpus} active | Up {data.get('uptime', '?')}")
        if data.get("rate"):
            lines.append(f":chart_with_upwards_trend: {data['rate']}")
        return {"text": "\n".join(lines)}
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
    if event == "plateau":
        return {"text": f":warning: *Plateau detected*: {data.get('message', 'No improvement')}"}
    if event == "needs_intervention":
        role = data.get("role", "?")
        reason = data.get("reason", "unknown")
        evidence = str(data.get("evidence", ""))[:200]
        log_tail = str(data.get("log_tail", ""))[:500]
        host = data.get("host", socket.gethostname())
        pid = data.get("pid", "?")
        return {"text": (f":warning: *{role}* needs human intervention\n"
                         f"*Reason:* {reason}\n"
                         f"*Evidence:* {evidence}\n"
                         f"*Log tail:*\n```\n{log_tail}\n```\n"
                         f"*Host:* `{host}` (pid {pid})")}
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
    if not data.get("summary_only"):
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
        completed = data.get("completed", 0)
        failed = data.get("failed", 0)
        queued = data.get("queued", 0)
        running = data.get("running", data.get("training", 0) + data.get("eval", 0))
        total_gpus = data.get("total_gpus", running + data.get("free", 0))
        active_gpus = total_gpus - data.get("free", 0)
        lines = [
            f"\U0001f4ca **Orze Status** \u2014 `{host}`",
            (f"\u2705 {completed} completed | \u274c {failed} failed | "
             f"\u23f3 {queued} queued | \U0001f504 {running} running"),
        ]
        best_val = data.get("best_val")
        if best_val is not None:
            metric = data.get("best_metric", "score")
            best_id = data.get("best_id", "?")
            val_str = f"{best_val:.4f}" if isinstance(best_val, float) else str(best_val)
            lines.append(f"\U0001f3c6 Best: {val_str} {metric} (`{best_id}`)")
        lines.append(f"\U0001f4bb GPUs: {active_gpus}/{total_gpus} active | Up {data.get('uptime', '?')}")
        if data.get("rate"):
            lines.append(f"\U0001f4c8 {data['rate']}")
        return {"content": "\n".join(lines)}
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
    if event == "plateau":
        return {"content": f"\u26a0\ufe0f **Plateau detected**: {data.get('message', 'No improvement')}"}
    if event == "needs_intervention":
        role = data.get("role", "?")
        reason = data.get("reason", "unknown")
        evidence = str(data.get("evidence", ""))[:200]
        log_tail = str(data.get("log_tail", ""))[:500]
        host = data.get("host", socket.gethostname())
        pid = data.get("pid", "?")
        return {"content": (f"\u26a0\ufe0f **{role}** needs human intervention\n"
                            f"**Reason:** {reason}\n"
                            f"**Evidence:** {evidence}\n"
                            f"**Log tail:**\n```\n{log_tail}\n```\n"
                            f"**Host:** `{host}` (pid {pid})")}
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
    if not data.get("summary_only"):
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
        completed = data.get("completed", 0)
        failed = data.get("failed", 0)
        queued = data.get("queued", 0)
        running = data.get("running", data.get("training", 0) + data.get("eval", 0))
        total_gpus = data.get("total_gpus", running + data.get("free", 0))
        uptime = esc(str(data.get("uptime", "?")))
        active_runs = data.get("active_runs") or []

        # Stall detection: 0 completions + experiments running longer than timeout
        # Use configured timeout (default 6h) — stall means oldest experiment
        # exceeded the timeout, not just "hasn't finished yet".
        rate_str = str(data.get("rate", ""))
        rate_num = 0
        try:
            rate_num = int(rate_str.split()[0])
        except (ValueError, IndexError):
            pass
        max_elapsed = max((r.get("elapsed_min", 0) for r in active_runs),
                         default=0)
        stall_threshold = data.get("timeout", 21600) / 60  # seconds -> minutes
        is_stalled = (rate_num == 0 and active_runs
                      and max_elapsed >= stall_threshold)

        # Header with health status
        if is_stalled:
            lines = [
                f"\u26a0\ufe0f <b>Orze</b> \u2014 <code>{host}</code> \u2014 {uptime}"
                f" \u2014 <b>STALLED</b>",
            ]
        else:
            lines = [
                f"\U0001f4ca <b>Orze</b> \u2014 <code>{host}</code> \u2014 {uptime}",
            ]

        # Best result with target comparison
        best_val = data.get("best_val")
        est_label = data.get("estimate_label")
        metric_name = esc(str(data.get("best_metric", "score")))
        lower_is_better = data.get("sort", "descending") == "ascending"
        if best_val is not None:
            best_id = esc(str(data.get("best_id", "?")))
            if isinstance(best_val, float):
                val_str = f"{best_val:.2f}"
            else:
                val_str = str(best_val)
            tag = f" [{esc(est_label)}]" if est_label else ""
            target = data.get("target")
            if target is not None:
                beats_target = (best_val <= target if lower_is_better
                                else best_val >= target)
                if isinstance(best_val, (int, float)) and beats_target:
                    status = "\u2705"
                else:
                    gap = best_val - target if isinstance(best_val, (int, float)) else 0
                    status = f"\u274c {gap:+.2f}"
                lines.append(
                    f"\U0001f3c6 <b>{esc(val_str)}</b>{tag} {metric_name} "
                    f"(<code>{best_id}</code>) "
                    f"| target {target} {status}")
            else:
                lines.append(
                    f"\U0001f3c6 <b>{esc(val_str)}</b>{tag} {metric_name} "
                    f"(<code>{best_id}</code>)")

        # Verified results (full-scale) with per-dataset breakdown
        verified = data.get("verified")
        if verified:
            v_primary = verified.get("primary") or verified.get("avg_wer")
            if v_primary is not None:
                v_label = " [full]" if est_label else ""
                v_target = verified.get("target")
                gap_str = ""
                if v_target and isinstance(v_primary, (int, float)):
                    gap = v_primary - v_target
                    gap_str = f" | gap {gap:+.2f}"
                lines.append(
                    f"\U0001f3af Verified: <b>{v_primary:.2f}</b>"
                    f"{v_label}{gap_str}")
                per_ds = verified.get("per_dataset")
                if per_ds:
                    parts = [f"{esc(k)}={v:.1f}" for k, v
                             in sorted(per_ds.items())]
                    lines.append(f"  {' | '.join(parts)}")

        # Progress line with normalized rate and completion %
        rate = data.get("rate", "")
        hb_interval = data.get("heartbeat_interval", 3600)
        rate_display = esc(str(rate))
        if rate_num > 0 and hb_interval > 0:
            per_hr = rate_num / (hb_interval / 3600)
            rate_display += f" ({per_hr:.1f}/hr)"
        total_work = completed + failed + queued + running
        pct_str = ""
        if total_work > 0:
            pct = completed / total_work * 100
            pct_str = f" | {pct:.0f}% done"
        lines.append(
            f"\U0001f4c8 {completed} done | {failed} fail | "
            f"{queued} queued | {rate_display}{pct_str}")

        # GPU status — labeled clearly
        n_free = data.get("free", 0)
        n_training = data.get("training", 0)
        n_eval = data.get("eval", 0)
        eval_backlog = data.get("eval_backlog", 0)
        n_active = running
        if n_training and n_eval:
            gpu_line = (f"\U0001f5a5 {n_active} active "
                        f"({n_training} train, {n_eval} eval), "
                        f"{n_free} idle / {total_gpus} GPUs")
        elif n_active:
            gpu_line = (f"\U0001f5a5 {n_active} active, "
                        f"{n_free} idle / {total_gpus} GPUs")
        else:
            gpu_line = f"\U0001f5a5 {total_gpus} GPUs, all idle"
        if eval_backlog:
            gpu_line += f" | {eval_backlog} eval pending"
        lines.append(gpu_line)

        # Active experiments — compact (truncated IDs, count if many)
        if active_runs:
            if len(active_runs) > 4:
                oldest = max(r.get("elapsed_min", 0) for r in active_runs)
                newest = min(r.get("elapsed_min", 0) for r in active_runs)
                if oldest >= 60:
                    oldest_str = f"{oldest / 60:.1f}h"
                else:
                    oldest_str = f"{oldest:.0f}m"
                if newest >= 60:
                    newest_str = f"{newest / 60:.1f}h"
                else:
                    newest_str = f"{newest:.0f}m"
                lines.append(
                    f"\u23f3 {len(active_runs)} running "
                    f"({newest_str}\u2013{oldest_str})")
            else:
                run_parts = []
                for r in active_runs:
                    rid = esc(str(r.get("idea_id", "?")))
                    elapsed = r.get("elapsed_min", 0)
                    phase = r.get("phase", "")
                    if elapsed >= 60:
                        t = f"{elapsed / 60:.1f}h"
                    else:
                        t = f"{elapsed:.0f}m"
                    suffix = "/eval" if phase == "eval" else ""
                    run_parts.append(f"{rid}({t}{suffix})")
                lines.append(f"\u23f3 {' '.join(run_parts)}")

        # Stall warning detail
        if is_stalled:
            lines.append(
                f"\u26a0\ufe0f <b>0 completions with "
                f"{len(active_runs)} running ({max_elapsed:.0f}m oldest)</b>")

        # Next up in queue
        next_queue = data.get("next_queue") or []
        if next_queue:
            q_parts = []
            for q in next_queue:
                qid = esc(str(q.get("id", "?")))
                qtitle = esc(str(q.get("title", ""))[:30])
                q_parts.append(f"<code>{qid}</code> {qtitle}")
            lines.append(f"\U0001f4cb Next up:\n  " + "\n  ".join(q_parts))

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

    if event == "plateau":
        msg = esc(str(data.get("message", "No improvement")))
        text = f"\u26a0\ufe0f <b>Plateau detected</b>: {msg}"
        return url, {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}

    if event == "needs_intervention":
        role = esc(str(data.get("role", "?")))
        reason = esc(str(data.get("reason", "unknown")))
        evidence = esc(str(data.get("evidence", ""))[:200])
        log_tail = esc(str(data.get("log_tail", ""))[:500])
        host = esc(data.get("host", socket.gethostname()))
        pid = esc(str(data.get("pid", "?")))
        text = (f"\u26a0\ufe0f <b>{role}</b> needs human intervention\n"
                f"<b>Reason:</b> {reason}\n"
                f"<b>Evidence:</b> <code>{evidence}</code>\n"
                f"<b>Log tail:</b>\n<pre>{log_tail}</pre>\n"
                f"<b>Host:</b> <code>{host}</code> (pid {pid})")
        return url, {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}

    idea_id = esc(str(data.get("idea_id", "")))
    title = esc(str(data.get("title", "")))

    if event == "new_best":
        prev_id = esc(str(data.get("prev_best_id", "none")))
        metric = esc(str(data.get("metric_name", "")))
        val = data.get("metric_value", "")
        val_str = esc(str(val))
        # Show improvement delta
        prev_val = data.get("prev_best_val")
        delta_str = ""
        try:
            if prev_val is not None:
                curr_f = float(val)
                prev_f = float(prev_val)
                delta = curr_f - prev_f
                delta_str = f" | \u0394 {delta:+.2f}%"
        except (ValueError, TypeError):
            pass
        text = (f"\U0001f3c6\U0001f3c6\U0001f3c6 <b>NEW BEST</b>\n"
                f"<code>{idea_id}</code>: {title}\n"
                f"{metric}: <b>{val_str}</b>"
                f" (was {esc(str(prev_val or '?'))}% <code>{prev_id}</code>)"
                f"{delta_str}")
    elif event == "failed":
        err = esc(str(data.get("error") or "unknown")[:200])
        text = (f"\u274c <code>{idea_id}</code>: {title}\n"
                f"{err}")
    else:
        t_str = ""
        try:
            t = data.get("training_time")
            if t is not None:
                secs = float(t)
                if secs >= 3600:
                    t_str = f" | {secs / 3600:.1f}h"
                else:
                    t_str = f" | {secs / 60:.0f}m"
        except (ValueError, TypeError):
            pass
        metric = esc(str(data.get("metric_name", "")))
        val = esc(str(data.get("metric_value", "")))
        rank = data.get("rank", "?")
        rank_str = esc(str(rank))
        # Rank emoji
        if isinstance(rank, int) and rank <= 3:
            medal = ["\U0001f947", "\U0001f948", "\U0001f949"][rank - 1]
        else:
            medal = "\u2705"
        text = (f"{medal} <code>{idea_id}</code>: {title}\n"
                f"{metric}: <b>{val}</b> (#{rank_str}){t_str}")
    if not data.get("summary_only"):
        text += _format_leaderboard(data, lambda s: f"<b>{s}</b>", escape_fn=esc)

    return url, {"chat_id": chat_id, "text": text, "parse_mode": "HTML"}


def notify(event: str, data: dict, cfg: dict):
    """Send notifications for an event to all configured channels. Never raises."""
    try:
        ncfg = cfg.get("notifications") or {}
        if not ncfg.get("enabled", False):
            return

        logger.info("Sending notification for event: %s", event)

        # Suppress role_summary if configured
        if event == "role_summary" and ncfg.get("suppress_role_summary", False):
            logger.debug("Suppressing role_summary notification (configured)")
            return

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
        logger.error("Notification dispatch error: %s: %s", type(e).__name__, e)
