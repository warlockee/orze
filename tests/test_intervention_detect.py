"""Tests for intervention detection patterns."""
import pytest

from orze.engine.intervention_detect import detect


def test_detect_hf_gated():
    """Test hf_gated pattern detection."""
    log_tail = """
Starting model download from HuggingFace...
Error: You need to agree to share your contact information to access this model.
Failed to download weights.
"""
    result = detect(log_tail)
    assert result is not None
    code, line = result
    assert code == "hf_gated"
    assert "contact information" in line


def test_detect_hf_login():
    """Test hf_login pattern detection."""
    log_tail = """
Authentication required.
Please run: huggingface-cli login
"""
    result = detect(log_tail)
    assert result is not None
    code, line = result
    assert code == "hf_login"
    assert "huggingface-cli login" in line


def test_detect_hf_token_missing():
    """Test hf_token_missing pattern detection."""
    log_tail = """
Error: Token is required to access this resource.
Set HF_TOKEN environment variable.
"""
    result = detect(log_tail)
    assert result is not None
    code, _ = result
    assert code == "hf_token_missing"


def test_detect_gh_login():
    """Test gh_login pattern detection."""
    log_tail = """
GitHub authentication failed.
Please run: gh auth login
"""
    result = detect(log_tail)
    assert result is not None
    code, _ = result
    assert code == "gh_login"


def test_detect_openai_key():
    """Test openai_key pattern detection."""
    log_tail = """
Error initializing OpenAI client.
OPENAI_API_KEY is not set in environment.
"""
    result = detect(log_tail)
    assert result is not None
    code, _ = result
    assert code == "openai_key"


def test_detect_anthropic_key():
    """Test anthropic_key pattern detection."""
    log_tail = """
Failed to create Anthropic client.
ANTHROPIC_API_KEY not set.
"""
    result = detect(log_tail)
    assert result is not None
    code, _ = result
    assert code == "anthropic_key"


def test_detect_disk_full():
    """Test disk_full pattern detection."""
    log_tail = """
Writing checkpoint to disk...
OSError: [Errno 28] No space left on device
"""
    result = detect(log_tail)
    assert result is not None
    code, line = result
    assert code == "disk_full"
    assert "No space left on device" in line


def test_detect_oom():
    """Test oom pattern detection."""
    log_tail = """
Forward pass...
RuntimeError: CUDA out of memory. Tried to allocate 2.00 GiB
"""
    result = detect(log_tail)
    assert result is not None
    code, line = result
    assert code == "oom"
    assert "CUDA out of memory" in line


def test_detect_sudo_prompt():
    """Test sudo_prompt pattern detection."""
    log_tail = """
Installing system dependency...
sudo: a password is required
"""
    result = detect(log_tail)
    assert result is not None
    code, _ = result
    assert code == "sudo_prompt"


def test_detect_unrelated_log_returns_none():
    """Test that unrelated log returns None."""
    log_tail = """
Training epoch 1/10
Batch 100/1000 loss=0.5
Validation accuracy: 0.85
"""
    result = detect(log_tail)
    assert result is None


def test_detect_extra_patterns_merge():
    """Test that extra patterns merge correctly."""
    log_tail = """
Custom error occurred.
CUSTOM_VAR is not set.
"""
    extra = {"custom_error": [r"CUSTOM_VAR"]}
    result = detect(log_tail, extra_patterns=extra)
    assert result is not None
    code, _ = result
    assert code == "custom_error"
