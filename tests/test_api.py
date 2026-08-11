import math
import json

import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.model_registry import get_models_root
from pipelines.ingestion.load_alerts import load_raw_alerts

REGISTRY_AVAILABLE = (get_models_root() / "entity" / "scoring_registry.json").exists() and (
    get_models_root() / "transaction" / "scoring_registry.json"
).exists()


@pytest.fixture
def client():
    return TestClient(app)


def _real_payload(alert_type: str, sheet_name: str) -> dict:
    from app.services.scoring import allowed_request_fields

    sheets = load_raw_alerts()
    row = sheets[sheet_name].iloc[1]
    fields = allowed_request_fields(alert_type)
    out = {}
    for c in fields:
        v = row[c]
        if isinstance(v, float) and math.isnan(v):
            v = None
        elif hasattr(v, "item"):
            v = v.item()
        elif hasattr(v, "isoformat"):
            v = v.isoformat()
        out[c] = v
    return out


def test_root_endpoint(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "health" in r.json()


def test_health_endpoint_shape(client):
    r = client.get("/api/v1/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] in ("ok", "degraded")
    assert set(body["models_loaded"].keys()) == {"customer_name", "transaction_name", "transaction_rule"}


def test_score_endpoint_rejects_unknown_alert_type(client):
    r = client.post("/api/v1/alerts/score", json={
        "alert_id": "x", "alert_type": "not_a_real_type", "raw_fields": {},
    })
    assert r.status_code == 422  # pydantic Literal validation rejects it before it reaches our code


@pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="requires scoring registry")
def test_score_endpoint_missing_required_field_returns_422(client):
    r = client.post("/api/v1/alerts/score", json={
        "alert_id": "x", "alert_type": "customer_name", "raw_fields": {"UIN": 1},
    })
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert len(detail["missing_required_columns"]) > 0


@pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="requires scoring registry")
@pytest.mark.parametrize(
    "alert_type,sheet_name",
    [("customer_name", "CustomerViolation"), ("transaction_name", "TransactionNameViolation"), ("transaction_rule", "Rule")],
)
def test_score_endpoint_real_alert_all_types(client, alert_type, sheet_name):
    raw = _real_payload(alert_type, sheet_name)
    r = client.post("/api/v1/alerts/score", json={
        "alert_id": "api-test", "alert_type": alert_type, "raw_fields": raw,
    })
    assert r.status_code == 200
    body = r.json()
    assert body["alert_id"] == "api-test"
    assert 0.0 <= body["novelty"]["global"] <= 100.0
    assert body["recommendation"] in ("REVIEW", "LOWER_TOUCH_CANDIDATE")
    # never claim probability -- the note must be present and explicit
    assert "not a probability" in body["novelty_scale_note"].lower() or "NOT a probability" in body["novelty_scale_note"]


@pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="requires scoring registry")
def test_batch_score_endpoint_partial_success(client):
    good = _real_payload("customer_name", "CustomerViolation")
    r = client.post("/api/v1/alerts/batch-score", json={"alerts": [
        {"alert_id": "good-1", "alert_type": "customer_name", "raw_fields": good},
        {"alert_id": "bad-1", "alert_type": "customer_name", "raw_fields": {"UIN": 1}},
    ]})
    assert r.status_code == 200
    body = r.json()
    assert body["n_requested"] == 2
    assert body["n_scored"] == 1
    assert body["n_failed"] == 1
    assert body["results"][0]["alert_id"] == "good-1"
    assert body["errors"][0]["alert_id"] == "bad-1"


@pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="requires scoring registry")
def test_models_active_endpoint(client):
    r = client.get("/api/v1/models/active")
    assert r.status_code == 200
    body = r.json()
    alert_types = {m["alert_type"] for m in body["models"]}
    assert alert_types == {"customer_name", "transaction_name", "transaction_rule"}


@pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="requires scoring registry")
def test_model_by_version_found_and_not_found(client):
    active = client.get("/api/v1/models/active").json()
    version = active["models"][0]["model_version"]

    r_found = client.get(f"/api/v1/models/{version}")
    assert r_found.status_code == 200
    assert r_found.json()["model_version"] == version

    r_missing = client.get("/api/v1/models/definitely-not-a-real-version")
    assert r_missing.status_code == 404


def test_feedback_endpoint_writes_entry(client, tmp_path, monkeypatch):
    feedback_path = tmp_path / "feedback_log.jsonl"
    monkeypatch.setenv("FEEDBACK_STORE_PATH", str(feedback_path))

    r = client.post("/api/v1/feedback", json={
        "alert_id": "fb-1", "alert_type": "customer_name", "model_version": "v-test",
        "compliance_outcome": "UNKNOWN", "notes": "test note",
    })
    assert r.status_code == 200
    body = r.json()
    assert body["received"] is True
    assert body["alert_id"] == "fb-1"

    assert feedback_path.exists()
    lines = feedback_path.read_text().strip().split("\n")
    assert len(lines) == 1
    entry = json.loads(lines[0])
    assert entry["alert_id"] == "fb-1"
    assert entry["compliance_outcome"] == "UNKNOWN"


def test_feedback_endpoint_rejects_invalid_outcome(client, tmp_path, monkeypatch):
    monkeypatch.setenv("FEEDBACK_STORE_PATH", str(tmp_path / "fb.jsonl"))
    r = client.post("/api/v1/feedback", json={
        "alert_id": "fb-2", "alert_type": "customer_name", "model_version": "v-test",
        "compliance_outcome": "DEFINITELY_TRUE_MATCH",  # not in the allowed taxonomy
    })
    assert r.status_code == 422
