import json

import pytest

from app.services.shadow_log import (
    get_shadow_log_path,
    read_shadow_log,
    record_shadow_score,
    verify_no_autonomous_action,
)


def _fake_score_result(alert_id="a1"):
    return {
        "alert_id": alert_id,
        "alert_type": "customer_name",
        "model_version": "v1",
        "feature_version": "fv1",
        "schema_version": "sv1",
        "novelty": {"global": 42.0, "customer": 0.5},
        "recommendation": "REVIEW",
        "plain_language_label": "Needs Review",
        "scored_at": "2026-01-01T00:00:00+00:00",
    }


def test_shadow_log_path_configurable(monkeypatch, tmp_path):
    custom = tmp_path / "custom_shadow.jsonl"
    monkeypatch.setenv("SHADOW_LOG_PATH", str(custom))
    assert get_shadow_log_path() == custom


def test_record_shadow_score_always_sets_autonomous_action_false(monkeypatch, tmp_path):
    monkeypatch.setenv("SHADOW_LOG_PATH", str(tmp_path / "shadow.jsonl"))
    entry = record_shadow_score(_fake_score_result(), source="test")
    assert entry["autonomous_action_taken"] is False
    assert entry["shadow_mode"] is True


def test_record_shadow_score_appends_not_overwrites(monkeypatch, tmp_path):
    monkeypatch.setenv("SHADOW_LOG_PATH", str(tmp_path / "shadow.jsonl"))
    record_shadow_score(_fake_score_result("a1"), source="test")
    record_shadow_score(_fake_score_result("a2"), source="test")
    entries = read_shadow_log()
    assert len(entries) == 2
    assert [e["alert_id"] for e in entries] == ["a1", "a2"]


def test_read_shadow_log_empty_when_no_file(monkeypatch, tmp_path):
    monkeypatch.setenv("SHADOW_LOG_PATH", str(tmp_path / "does_not_exist.jsonl"))
    assert read_shadow_log() == []


def test_verify_no_autonomous_action_clean_log(monkeypatch, tmp_path):
    monkeypatch.setenv("SHADOW_LOG_PATH", str(tmp_path / "shadow.jsonl"))
    record_shadow_score(_fake_score_result("a1"), source="test")
    record_shadow_score(_fake_score_result("a2"), source="test")
    ok, violations = verify_no_autonomous_action()
    assert ok
    assert violations == []


def test_verify_no_autonomous_action_detects_violation(tmp_path):
    """The invariant check must actually scan the file content, not trust
    the writer -- prove it by hand-crafting a violating record."""
    path = tmp_path / "shadow.jsonl"
    with open(path, "w") as f:
        f.write(json.dumps({"alert_id": "bad-1", "autonomous_action_taken": True}) + "\n")
        f.write(json.dumps({"alert_id": "good-1", "autonomous_action_taken": False}) + "\n")

    ok, violations = verify_no_autonomous_action(path)
    assert not ok
    assert violations == ["bad-1"]


def test_shadow_log_entry_never_contains_raw_fields(monkeypatch, tmp_path):
    """Structural check: the audit log stores scores/metadata only, never
    the raw_fields dict a caller submitted (which may contain PII)."""
    monkeypatch.setenv("SHADOW_LOG_PATH", str(tmp_path / "shadow.jsonl"))
    entry = record_shadow_score(_fake_score_result(), source="test")
    assert "raw_fields" not in entry
