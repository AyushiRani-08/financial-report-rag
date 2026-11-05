"""Unit tests for judge request construction — no network calls."""

import importlib.util
import json
import types
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_spec = importlib.util.spec_from_file_location("run_eval", ROOT / "eval" / "run_eval.py")
run_eval = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(run_eval)


class _FakeCompletions:
    def __init__(self, content: str):
        self.content = content
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        message = types.SimpleNamespace(content=self.content)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=message)])


def _client(content='{"faithfulness": 5, "relevance": 5, "retail_clarity": 5, '
                    '"has_hallucination": false, "reasoning": "ok"}'):
    completions = _FakeCompletions(content)
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=completions)), completions


SOURCES = [{"text": "Apple net sales were $391.0 billion.", "metadata": {"page_number": 3}}]


def test_default_judge_stays_under_otpm(monkeypatch):
    monkeypatch.delenv("EVAL_JUDGE_MODEL", raising=False)
    client, completions = _client()
    run_eval.judge_answer(client, "What were net sales?", "Net sales were $391.0 billion.", SOURCES)
    call = completions.calls[0]
    assert call["model"] == "qwen/qwen3.8-27b"
    assert call["max_tokens"] == 800
    assert "reasoning_effort" not in call


def test_reasoning_judge_keeps_headroom(monkeypatch):
    monkeypatch.setenv("EVAL_JUDGE_MODEL", "openai/gpt-oss-20b")
    client, completions = _client()
    run_eval.judge_answer(client, "What were net sales?", "Net sales were $391.0 billion.", SOURCES)
    call = completions.calls[0]
    assert call["max_tokens"] == 1200
    assert call["reasoning_effort"] == "low"


def test_verified_xbrl_facts_reach_the_prompt():
    client, completions = _client()
    facts = "ticker=AAPL fiscal_period=FY2024 revenue=391,035M"
    run_eval.judge_answer(client, "What were net sales?", "Net sales were $391.0 billion.",
                          SOURCES, verified_facts=facts)
    prompt = completions.calls[0]["messages"][0]["content"]
    assert "[Verified XBRL facts" in prompt
    assert "revenue=391,035M" in prompt


def test_no_xbrl_section_when_facts_empty():
    client, completions = _client()
    run_eval.judge_answer(client, "What were net sales?", "Net sales were $391.0 billion.",
                          SOURCES, verified_facts="")
    prompt = completions.calls[0]["messages"][0]["content"]
    assert "[Verified XBRL facts" not in prompt


def test_judge_returns_parsed_scores():
    client, _ = _client()
    scores = run_eval.judge_answer(client, "q", "a", SOURCES)
    assert scores["faithfulness"] == 5
    assert scores["has_hallucination"] is False
    assert scores["judge_used_fallback"] is False


def test_judge_fails_closed_on_garbage_output(monkeypatch):
    monkeypatch.setattr(run_eval.time, "sleep", lambda *_: None)
    client, _ = _client(content="I cannot grade this.")
    scores = run_eval.judge_answer(client, "q", "a", SOURCES)
    assert scores["faithfulness"] is None
    assert scores["judge_used_fallback"] is True


if __name__ == "__main__":
    import pytest
    raise SystemExit(pytest.main([__file__, "-q"]))
