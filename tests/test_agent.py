"""Tests for agent module."""

import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import GROQ_MAX_COMPLETION_TOKENS
from agent import (
    FIX_GENERATION_SYSTEM_PROMPT,
    INVESTIGATION_SYSTEM_PROMPT,
    TOOL_DEFINITIONS,
    call_llm,
    parse_agent_action,
)


# ── 1. parse_agent_action ────────────────────────────────

def test_parse_agent_action_tool_call_prefix():
    """Test parsing TOOL_CALL with explicit prefix."""
    raw = 'TOOL_CALL: {"name": "read_file", "arguments": {"filepath": "src/main.py"}}'
    action = parse_agent_action(raw)
    assert action["type"] == "tool_call"
    assert action["name"] == "read_file"
    assert action["arguments"] == {"filepath": "src/main.py"}


def test_parse_agent_action_json_block_extraction():
    """Test extracting tool call from embedded JSON without prefix."""
    raw = 'I will inspect the file now:\n{"name": "search_code", "arguments": {"pattern": "def hello"}}'
    action = parse_agent_action(raw)
    assert action["type"] == "tool_call"
    assert action["name"] == "search_code"
    assert action["arguments"] == {"pattern": "def hello"}


def test_parse_agent_action_final_answer():
    """Test parsing explicit FINAL_ANSWER response."""
    raw = "FINAL_ANSWER: The bug was caused by a missing import statement in src/app.py."
    action = parse_agent_action(raw)
    assert action["type"] == "final_answer"
    assert "missing import statement" in action["text"]


def test_parse_agent_action_long_text_fallback():
    """Test that long plain-text responses without tool markers fall back to final_answer."""
    raw = "This is an extensive analysis of the system architecture without issuing any tool invocations. " * 3
    action = parse_agent_action(raw)
    assert action["type"] == "final_answer"
    assert action["text"] == raw.strip()


def test_parse_agent_action_malformed():
    """Test handling malformed or unparseable short responses."""
    raw = "Just random brief text."
    action = parse_agent_action(raw)
    assert action["type"] == "error"
    assert "could not parse" in action["text"].lower()


# ── 3. System Prompts ────────────────────────────────────

def test_system_prompts_non_empty():
    """Test that investigation and fix system prompts are valid non-empty strings."""
    assert isinstance(INVESTIGATION_SYSTEM_PROMPT, str)
    assert len(INVESTIGATION_SYSTEM_PROMPT.strip()) > 50

    assert isinstance(FIX_GENERATION_SYSTEM_PROMPT, str)
    assert len(FIX_GENERATION_SYSTEM_PROMPT.strip()) > 50


# ── 4. Tool Definitions ──────────────────────────────────

def test_tool_definitions_list():
    """Test that TOOL_DEFINITIONS is a non-empty list of tool specifications."""
    assert isinstance(TOOL_DEFINITIONS, list)
    assert len(TOOL_DEFINITIONS) > 0
    names = [t["name"] for t in TOOL_DEFINITIONS]
    assert "read_file" in names
    assert "list_files" in names
    assert "propose_patch" in names


# ── 5. Integration LLM Call ──────────────────────────────

@pytest.mark.integration
def test_call_llm_integration():
    """Test live call_llm when Ollama service is reachable."""
    from config import check_ollama_available
    if not check_ollama_available():
        pytest.skip("Ollama is not running.")

    result = call_llm("Return the word PONG", timeout=15)
    assert isinstance(result, dict)
    assert "success" in result
    assert "response" in result


def test_groq_gpt_oss_request_uses_supported_reasoning_settings(monkeypatch):
    import agent

    captured = {}

    class FakeResponse:
        status_code = 200
        headers = {}
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": "PONG"}}]}

    def fake_post(url, json, headers, timeout):
        captured.update(url=url, payload=json, headers=headers, timeout=timeout)
        return FakeResponse()

    monkeypatch.setattr(agent.requests, "post", fake_post)
    result = call_llm(
        "Return PONG", model="openai/gpt-oss-20b", provider="groq",
        api_key="test-key", base_url="https://api.groq.com/openai/v1"
    )

    assert result["success"] is True
    assert result["response"] == "PONG"
    assert captured["payload"]["model"] == "openai/gpt-oss-20b"
    assert captured["payload"]["reasoning_effort"] == "low"
    assert captured["payload"]["include_reasoning"] is False
    assert captured["payload"]["max_completion_tokens"] == GROQ_MAX_COMPLETION_TOKENS


def test_groq_gpt_oss_agent_request_declares_tools_and_converts_native_call(monkeypatch):
    import agent

    captured = {}

    class FakeResponse:
        status_code = 200
        headers = {}
        text = ""

        def json(self):
            return {"choices": [{"message": {
                "content": None,
                "tool_calls": [{"type": "function", "function": {
                    "name": "list_files", "arguments": '{"directory": "."}'
                }}]
            }}]}

    def fake_post(url, json, headers, timeout):
        captured.update(url=url, payload=json)
        return FakeResponse()

    monkeypatch.setattr(agent.requests, "post", fake_post)
    result = call_llm(
        "Inspect the project", model="openai/gpt-oss-20b", provider="groq",
        api_key="test-key", base_url="https://api.groq.com/openai/v1",
        enable_tools=True,
    )

    assert result["success"] is True
    assert result["response"] == 'TOOL_CALL: {"name": "list_files", "arguments": {"directory": "."}}'
    assert captured["payload"]["tool_choice"] == "auto"
    assert any(t["function"]["name"] == "list_files" for t in captured["payload"]["tools"])


def test_groq_gpt_oss_chat_request_does_not_enable_tools_by_default(monkeypatch):
    import agent
    captured = {}

    class FakeResponse:
        status_code = 200
        headers = {}
        text = ""

        def json(self):
            return {"choices": [{"message": {"content": "Hello!"}}]}

    def fake_post(url, json, headers, timeout):
        captured.update(payload=json)
        return FakeResponse()

    monkeypatch.setattr(agent.requests, "post", fake_post)
    result = call_llm(
        "Hello", model="openai/gpt-oss-20b", provider="groq",
        api_key="test-key", base_url="https://api.groq.com/openai/v1",
    )
    assert result["success"] is True
    assert "tools" not in captured["payload"]
    assert "tool_choice" not in captured["payload"]
