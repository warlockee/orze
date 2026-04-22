"""Tests for needs_intervention notification event formatters."""
import socket

from orze.reporting.notifications import _format_slack, _format_discord, _format_telegram


def test_slack_needs_intervention_format():
    """Test Slack formatter for needs_intervention event."""
    data = {
        "role": "engineer",
        "reason": "hf_gated",
        "evidence": "You need to agree to share your contact",
        "log_tail": "Error downloading model\nYou need to agree to share your contact information\nFailed.",
        "host": "gpu-node-01",
        "pid": 12345,
    }
    
    result = _format_slack("needs_intervention", data)
    
    assert "text" in result
    text = result["text"]
    assert "engineer" in text
    assert "hf_gated" in text
    assert "You need to agree to share your contact" in text
    assert "Error downloading model" in text
    assert "gpu-node-01" in text
    assert "12345" in text


def test_discord_needs_intervention_format():
    """Test Discord formatter for needs_intervention event."""
    data = {
        "role": "data_analyst",
        "reason": "disk_full",
        "evidence": "No space left on device",
        "log_tail": "Writing file...\nOSError: [Errno 28] No space left on device\n",
        "host": "node-02",
        "pid": 99999,
    }
    
    result = _format_discord("needs_intervention", data)
    
    assert "content" in result
    content = result["content"]
    assert "data_analyst" in content
    assert "disk_full" in content
    assert "No space left on device" in content
    assert "node-02" in content
    assert "99999" in content


def test_telegram_needs_intervention_format():
    """Test Telegram formatter for needs_intervention event with HTML escaping."""
    data = {
        "role": "professor",
        "reason": "openai_key",
        "evidence": "OPENAI_API_KEY not set",
        "log_tail": "<script>alert('xss')</script>\nOPENAI_API_KEY not set",
        "host": socket.gethostname(),
        "pid": 54321,
    }
    
    # _format_telegram returns (url, payload) tuple
    url, payload = _format_telegram("needs_intervention", data, {"bot_token": "fake_token", "chat_id": "12345"})
    
    assert "sendMessage" in url
    assert "text" in payload
    text = payload["text"]
    
    # Check content is present
    assert "professor" in text
    assert "openai_key" in text
    assert "OPENAI_API_KEY not set" in text
    assert str(54321) in text
    
    # Check HTML is escaped (no raw <script>)
    assert "<script>" not in text or "&lt;script&gt;" in text
    assert "parse_mode" in payload
    assert payload["parse_mode"] == "HTML"


def test_all_formatters_include_required_fields():
    """Test that all formatters include role, reason, evidence, log_tail."""
    data = {
        "role": "thinker",
        "reason": "oom",
        "evidence": "CUDA out of memory",
        "log_tail": "RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB",
        "host": "test-host",
        "pid": 11111,
    }
    
    # Slack
    slack_result = _format_slack("needs_intervention", data)
    slack_text = slack_result["text"]
    assert "thinker" in slack_text
    assert "oom" in slack_text
    assert "CUDA out of memory" in slack_text
    
    # Discord
    discord_result = _format_discord("needs_intervention", data)
    discord_text = discord_result["content"]
    assert "thinker" in discord_text
    assert "oom" in discord_text
    assert "CUDA out of memory" in discord_text
    
    # Telegram
    url, tg_payload = _format_telegram("needs_intervention", data, {"bot_token": "tk", "chat_id": "123"})
    tg_text = tg_payload["text"]
    assert "thinker" in tg_text
    assert "oom" in tg_text
    assert "CUDA out of memory" in tg_text
