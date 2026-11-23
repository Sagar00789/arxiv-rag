"""Retry, quota and resume behaviour, tested against a fake `litellm` module."""

import json
import sys
import types

import pytest

from arxiv_rag import generate
from arxiv_rag.chunking import Chunk
from arxiv_rag.evaluation import load_testset, make_testset

RATE_LIMIT_MSG = "Quota exceeded ... GenerateRequestsPerMinute ... Please retry in 6.88s."
DAILY_MSG = "Quota exceeded ... GenerateRequestsPerDayPerProjectPerModel-FreeTier"


def fake_litellm(script):
    """A stand-in litellm whose completion() replays `script` (exceptions or reply text)."""
    mod = types.ModuleType("litellm")
    for name in (
        "NotFoundError",
        "BadRequestError",
        "AuthenticationError",
        "RateLimitError",
        "APIConnectionError",
    ):
        setattr(mod, name, type(name, (Exception,), {}))
    calls = []

    def completion(**kwargs):
        calls.append(kwargs)
        step = script.pop(0)
        if isinstance(step, tuple):  # (exception name, message)
            raise getattr(mod, step[0])(step[1])
        msg = types.SimpleNamespace(content=step)
        return types.SimpleNamespace(choices=[types.SimpleNamespace(message=msg)])

    mod.completion = completion
    mod.calls = calls
    return mod


@pytest.fixture
def sleeps(monkeypatch):
    waited = []
    monkeypatch.setattr(generate.time, "sleep", waited.append)
    generate.set_rate_limit(0)
    return waited


def install(monkeypatch, script):
    mod = fake_litellm(script)
    monkeypatch.setitem(sys.modules, "litellm", mod)
    return mod


def test_rate_limit_waits_as_told_then_succeeds(monkeypatch, sleeps):
    install(monkeypatch, [("RateLimitError", RATE_LIMIT_MSG), "hello"])
    assert generate.complete([{"role": "user", "content": "hi"}], model="m") == "hello"
    assert sleeps == [pytest.approx(7.88)]  # provider's 6.88s + 1s margin


def test_daily_quota_stops_immediately(monkeypatch, sleeps):
    mod = install(monkeypatch, [("RateLimitError", DAILY_MSG)])
    with pytest.raises(generate.LLMError, match="Daily"):
        generate.complete([{"role": "user", "content": "hi"}], model="m")
    assert len(mod.calls) == 1 and sleeps == []


def test_gives_up_after_max_retries(monkeypatch, sleeps):
    install(monkeypatch, [("APIConnectionError", "down")] * 3)
    with pytest.raises(generate.LLMError, match="after 2 retries"):
        generate.complete([{"role": "user", "content": "hi"}], model="m", max_retries=2)


def test_pacing_spaces_out_calls(monkeypatch, sleeps):
    install(monkeypatch, ["a", "b"])
    generate.set_rate_limit(12)  # one call per 5s
    generate.complete([{"role": "user", "content": "1"}], model="m")
    generate.complete([{"role": "user", "content": "2"}], model="m")
    generate.set_rate_limit(0)
    assert sleeps and sleeps[-1] == pytest.approx(5.0, abs=0.5)


def _chunks(n):
    return [Chunk(f"p::{i}", "p", "T", "word " * 150, i) for i in range(n)]


def q(i):
    return json.dumps({"question": f"q{i}?", "answer": "a"})


def test_testset_survives_failure_and_resumes(monkeypatch, sleeps, tmp_path):
    out = tmp_path / "testset.jsonl"
    # First run: 2 questions, then the daily quota runs out.
    install(monkeypatch, [q(1), q(2), ("RateLimitError", DAILY_MSG)])
    with pytest.raises(generate.LLMError):
        make_testset(_chunks(10), 4, "m", out)
    assert len(load_testset(out)) == 2  # nothing lost

    # Second run picks up where it stopped, on new chunks only.
    install(monkeypatch, [q(3), q(4)])
    items = make_testset(_chunks(10), 4, "m", out)
    assert len(items) == 4 == len(load_testset(out))
    assert len({it.gold_chunk_id for it in items}) == 4
