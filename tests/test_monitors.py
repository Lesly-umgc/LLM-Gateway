"""Tests for monitors/faithfulness.py.

Fast tests force the heuristic fallback (monkeypatched scorer) so they run
without Ollama. Tests marked `slow` need a live judge: Ollama up with the
LNM_JUDGE_MODEL pulled (default llama3.1:8b).
"""

import os
import urllib.request

import pytest

from monitors import faithfulness as fm

CTX = ("The Eiffel Tower is 330 meters tall and was completed in 1889. "
       "It is located in Paris, France.")
FAITHFUL = "The Eiffel Tower is 330 meters tall and was completed in 1889."
UNFAITHFUL = "The Eiffel Tower is 500 meters tall and was completed in 1901."
QUESTION = "How tall is the Eiffel Tower and when was it completed?"


@pytest.fixture(autouse=True)
def _reset_scorer():
    fm._scorer = None
    yield
    fm._scorer = None


def _judge_down(monkeypatch):
    def _raise():
        raise RuntimeError("judge unreachable")
    monkeypatch.setattr(fm, "_get_scorer", _raise)


def test_heuristic_faithful_scores_high(monkeypatch):
    _judge_down(monkeypatch)
    res = fm.faithfulness_score(FAITHFUL, CTX)
    assert res["judge"] == "heuristic"
    assert res["score"] >= 0.9
    assert res["flagged"] is False


def test_heuristic_unfaithful_flagged(monkeypatch):
    _judge_down(monkeypatch)
    # word-overlap can't catch subtle number swaps; it flags answers whose
    # content barely appears in the context at all
    res = fm.faithfulness_score("Penguins live in Antarctica and eat fish.",
                                CTX)
    assert res["judge"] == "heuristic"
    assert res["score"] < 0.7
    assert res["flagged"] is True


def test_heuristic_no_claims_scores_one(monkeypatch):
    _judge_down(monkeypatch)
    res = fm.faithfulness_score("Hello there!", CTX)
    assert res["score"] == 1.0
    assert res["claims"] == 0
    assert res["flagged"] is False


def test_monitor_plumbing_and_stats(monkeypatch):
    _judge_down(monkeypatch)
    seen = []
    mon = fm.FaithfulnessMonitor(sample_rate=1.0, on_scored=seen.append)
    mon.maybe_check(request_id="r1", answer=FAITHFUL, context=CTX, force=True)
    mon._q.join()
    assert len(seen) == 1
    assert seen[0]["request_id"] == "r1"
    assert seen[0]["score"] >= 0.9
    stats = mon.stats()
    assert stats["scored"] == 1
    assert stats["flagged"] == 0
    assert stats["mean_score"] is not None
    assert stats["min_score"] is not None


def test_monitor_skips_without_context():
    mon = fm.FaithfulnessMonitor(sample_rate=1.0)
    mon.maybe_check(request_id="r2", answer=FAITHFUL, context="", force=True)
    assert mon._q.qsize() == 0
    assert mon.stats()["scored"] == 0


def _ollama_up() -> bool:
    base = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")
    try:
        urllib.request.urlopen(base + "/api/tags", timeout=3)
        return True
    except Exception:
        return False


needs_judge = pytest.mark.skipif(not _ollama_up(),
                                 reason="needs Ollama with the judge model")


@pytest.mark.slow
@needs_judge
def test_ragas_faithful_answer():
    res = fm.faithfulness_score(FAITHFUL, CTX, question=QUESTION)
    assert res["judge"] == "ragas", res
    assert res["score"] >= 0.9, res


@pytest.mark.slow
@needs_judge
def test_ragas_unfaithful_answer_flagged():
    res = fm.faithfulness_score(UNFAITHFUL, CTX, question=QUESTION)
    assert res["judge"] == "ragas", res
    assert res["score"] < 0.7, res
    assert res["flagged"] is True
