import pytest
from fastapi.testclient import TestClient

from app.main import app
from app.services.model_registry import get_models_root
from pipelines.ingestion.load_alerts import get_raw_data_path

RAW_DATA_AVAILABLE = get_raw_data_path().exists()
REGISTRY_AVAILABLE = (get_models_root() / "entity" / "scoring_registry.json").exists() and (
    get_models_root() / "transaction" / "scoring_registry.json"
).exists()
DEMO_READY = RAW_DATA_AVAILABLE and REGISTRY_AVAILABLE


@pytest.fixture
def client():
    return TestClient(app)


def test_demo_page_serves_html(client):
    r = client.get("/demo")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Alert Intelligence Engine" in r.text


def test_static_assets_served(client):
    assert client.get("/static/demo.css").status_code == 200
    assert client.get("/static/demo.js").status_code == 200


def test_data_overview_endpoint(client):
    r = client.get("/api/v1/demo/data-overview")
    assert r.status_code == 200
    body = r.json()
    assert len(body["sheets"]) == 3


def test_sample_alert_endpoint_bad_alert_type(client):
    r = client.get("/api/v1/demo/sample-alert?alert_type=bogus")
    assert r.status_code == 400


def test_review_queue_endpoint_bad_alert_type(client):
    r = client.get("/api/v1/demo/review-queue?alert_type=bogus")
    assert r.status_code == 400


@pytest.mark.skipif(not DEMO_READY, reason="requires local raw data + scoring registry")
def test_sample_alert_endpoint_success(client):
    r = client.get("/api/v1/demo/sample-alert?alert_type=customer_name&seed=1")
    assert r.status_code == 200
    body = r.json()
    assert body["alert_type"] == "customer_name"
    assert body["raw_fields"]


@pytest.mark.skipif(not DEMO_READY, reason="requires local raw data + scoring registry")
def test_review_queue_endpoint_success(client):
    r = client.get("/api/v1/demo/review-queue?alert_type=transaction_name&n=5&seed=1")
    assert r.status_code == 200
    body = r.json()
    assert body["n"] == 5
    assert len(body["queue"]) == 5


@pytest.mark.skipif(not DEMO_READY, reason="requires local raw data + scoring registry")
def test_full_demo_flow_sample_then_score(client):
    """End-to-end: exactly what the UI does -- load a sample, then send it
    through the real production scoring endpoint."""
    sample_r = client.get("/api/v1/demo/sample-alert?alert_type=transaction_rule&seed=2")
    assert sample_r.status_code == 200
    sample = sample_r.json()

    score_r = client.post("/api/v1/alerts/score", json={
        "alert_id": sample["alert_id"],
        "alert_type": sample["alert_type"],
        "raw_fields": sample["raw_fields"],
    })
    assert score_r.status_code == 200
    result = score_r.json()
    assert result["alert_id"] == sample["alert_id"]
