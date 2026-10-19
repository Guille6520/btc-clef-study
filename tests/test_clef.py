"""Cliente de Clef: versiones de caché, preguntas, parseo y manejo de errores. Sin red."""
import json

import pytest

from clefbt import clef

CFG = {
    "labeling": {"horizon_hours": 24, "main_sl_atr": 1.5, "main_tp_atr": 3.0,
                 "grid_sl": [1.0, 1.5, 2.0], "grid_tp": [1.5, 2.0, 3.0]},
    "clef": {"model": "clef-flash", "question_grid": "main", "state_version": "v1", "timeout_s": 5},
}


def _cfg(**clef_overrides):
    return {**CFG, "clef": {**CFG["clef"], **clef_overrides}}


def test_qset_versions_are_stable_so_cached_calls_stay_valid():
    assert clef.qset_version(CFG) == "q1-main-h24-sl1.5-tp3.0"
    assert clef.qset_version(_cfg(question_set="dir")) == "dir1-h24"


def test_tpsl_questions_describe_the_configured_barriers():
    qs = clef.build_questions(CFG)
    assert set(qs) == {"long_sl1p5_tp3p0", "short_sl1p5_tp3p0"} == set(clef.main_keys(CFG).values())
    for q in qs.values():
        assert q["type"] == "noul"
        assert "1.5 ATR" in q["instructions"] and "3.0 ATR" in q["instructions"] and "24 hours" in q["instructions"]


def test_full_grid_has_eighteen_questions():
    assert len(clef.build_questions(_cfg(question_grid="full"))) == 18


def test_direction_question_set():
    qs = clef.build_questions(_cfg(question_set="dir"))
    assert list(qs) == [clef.DIR_KEY]
    assert "HIGHER" in qs[clef.DIR_KEY]["instructions"]


def test_parse_probs_reads_the_real_response_shape():
    resp = {"result": {"model": "clef-flash",
                       "answers": {"a": {"type": "noul", "noul": 0.1541}, "b": {"type": "noul", "noul": 0.4465}},
                       "usage": {"input_tokens": 739, "output_tokens": 0}},
            "success": True, "errors": [], "messages": []}
    assert clef.parse_probs(resp, {"a": {}, "b": {}}) == {"a": 0.1541, "b": 0.4465}


def test_parse_probs_fails_loudly_on_missing_question():
    with pytest.raises(ValueError):
        clef.parse_probs({"result": {"answers": {"a": {"noul": 0.5}}}}, {"a": {}, "b": {}})


class FakeResponse:
    def __init__(self, status, body):
        self.status_code = status
        self._body = body
        self.text = json.dumps(body)

    @property
    def ok(self):
        return self.status_code < 400

    def json(self):
        return self._body


@pytest.fixture
def env(monkeypatch):
    monkeypatch.setenv("CLOUDFLARE_ACCOUNT_ID", "a" * 32)
    monkeypatch.setenv("CLOUDFLARE_API_TOKEN", "secret-token-123")
    monkeypatch.setattr(clef.time, "sleep", lambda s: None)


def _patch_post(monkeypatch, responses):
    calls = []

    def fake_post(url, headers=None, json=None, timeout=None):
        calls.append(url)
        return responses[min(len(calls) - 1, len(responses) - 1)]

    monkeypatch.setattr(clef.requests, "post", fake_post)
    return calls


def test_success_returns_json(env, monkeypatch):
    _patch_post(monkeypatch, [FakeResponse(200, {"success": True, "result": {"answers": {}}})])
    assert clef.call_clef(CFG, "state", {})["success"] is True


def test_quota_exhausted_stops_immediately_without_retrying(env, monkeypatch):
    body = {"errors": [{"code": 4006, "message": "you have used up your daily free allocation of 10,000 neurons"}],
            "success": False}
    calls = _patch_post(monkeypatch, [FakeResponse(429, body)])
    with pytest.raises(clef.QuotaExceeded) as exc:
        clef.call_clef(CFG, "state", {})
    assert len(calls) == 1
    assert "secret-token-123" not in str(exc.value)          # el token nunca aparece en los errores


def test_transient_rate_limit_is_retried_then_gives_up(env, monkeypatch):
    body = {"errors": [{"code": 1015, "message": "rate limited"}], "success": False}
    calls = _patch_post(monkeypatch, [FakeResponse(429, body)])
    with pytest.raises(RuntimeError, match="tras reintentos"):
        clef.call_clef(CFG, "state", {})
    assert len(calls) == 5


def test_client_errors_are_not_retried(env, monkeypatch):
    calls = _patch_post(monkeypatch, [FakeResponse(401, {"errors": [{"code": 10000, "message": "Authentication error"}]})])
    with pytest.raises(RuntimeError, match="no reintentable") as exc:
        clef.call_clef(CFG, "state", {})
    assert len(calls) == 1
    assert "secret-token-123" not in str(exc.value)
