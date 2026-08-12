import pytest

from app.services.demo_data import ALERT_TYPE_TO_SHEET, build_review_queue, load_data_overview, sample_alert
from app.services.model_registry import get_models_root
from pipelines.ingestion.load_alerts import get_raw_data_path

RAW_DATA_AVAILABLE = get_raw_data_path().exists()
REGISTRY_AVAILABLE = (get_models_root() / "entity" / "scoring_registry.json").exists() and (
    get_models_root() / "transaction" / "scoring_registry.json"
).exists()
DEMO_READY = RAW_DATA_AVAILABLE and REGISTRY_AVAILABLE


def test_data_overview_shape_and_no_pii_risk():
    """This one has no skip guard -- it only reads committed, aggregate-only
    evaluation reports, so it must work in any environment."""
    overview = load_data_overview()
    assert len(overview["sheets"]) == 3
    assert overview["total_rows"] == 6641
    assert "label_audit_finding" in overview
    # sanity: nothing that looks like a name field leaked into this aggregate view
    assert "Alerted Party Name" not in str(overview)


@pytest.mark.skipif(not DEMO_READY, reason="requires local raw data + scoring registry")
@pytest.mark.parametrize("alert_type", list(ALERT_TYPE_TO_SHEET.keys()))
def test_sample_alert_returns_scoreable_payload(alert_type):
    from app.services.scoring import score_alert

    sample = sample_alert(alert_type, seed=1)
    assert sample["alert_type"] == alert_type
    assert sample["raw_fields"]
    assert "historical_operational_outcome" in sample
    # historical outcome must NOT be inside raw_fields (would be leakage if fed back to scoring)
    assert "Status" not in sample["raw_fields"]
    assert "Alert Status" not in sample["raw_fields"]

    # the sample must actually be scoreable through the real API path
    result = score_alert(alert_type, sample["alert_id"], sample["raw_fields"])
    assert 0.0 <= result["novelty"]["global"] <= 100.0


@pytest.mark.skipif(not DEMO_READY, reason="requires local raw data + scoring registry")
def test_sample_alert_is_from_the_unseen_test_split():
    """The sample must come from the same test split Phase 5/7 proved has
    zero customer overlap with training -- not an arbitrary row."""
    from pipelines.entity.combined_dataset import build_combined_entity_dataset
    from pipelines.entity.validation_splits import group_split_by_customer
    from pipelines.normalization.pipeline import run_phase2_pipeline
    from pathlib import Path

    normalized_sheets, _ = run_phase2_pipeline(Path.cwd(), persist=False)
    combined = build_combined_entity_dataset(
        normalized_sheets["CustomerViolation"], normalized_sheets["TransactionNameViolation"]
    )
    train_idx, _ = group_split_by_customer(combined, test_size=0.25, random_state=42)
    train_customer_ids = set(combined.iloc[train_idx]["customer_id"])

    sample = sample_alert("customer_name", seed=1)
    sample_uin = sample["raw_fields"]["UIN"]
    sample_customer_id = f"customerviolation:customer:{sample_uin}"
    assert sample_customer_id not in train_customer_ids


@pytest.mark.skipif(not DEMO_READY, reason="requires local raw data + scoring registry")
def test_review_queue_sorted_descending_by_global_novelty():
    queue = build_review_queue("transaction_rule", n=8, seed=1)
    scores = [r["novelty"]["global"] for r in queue]
    assert scores == sorted(scores, reverse=True)


@pytest.mark.skipif(not DEMO_READY, reason="requires local raw data + scoring registry")
def test_review_queue_deterministic_for_fixed_seed():
    q1 = build_review_queue("customer_name", n=5, seed=7)
    q2 = build_review_queue("customer_name", n=5, seed=7)
    assert [r["alert_id"] for r in q1] == [r["alert_id"] for r in q2]


def test_sample_alert_unknown_alert_type_raises():
    with pytest.raises(KeyError):
        sample_alert("not_a_real_type")


@pytest.mark.skipif(not DEMO_READY, reason="requires local raw data + scoring registry")
def test_review_queue_includes_identity_block_for_entity_alert():
    queue = build_review_queue("customer_name", n=3, seed=1)
    for r in queue:
        assert "identity" in r
        assert r["identity"]["name"] is not None
        assert r["identity"]["id"] is not None


@pytest.mark.skipif(not DEMO_READY, reason="requires local raw data + scoring registry")
def test_review_queue_rule_identity_has_no_dob_field():
    """Rule sheet has no per-customer DOB field -- must be None, not a
    KeyError or a wrong/fabricated value."""
    queue = build_review_queue("transaction_rule", n=3, seed=1)
    for r in queue:
        assert r["identity"]["dob"] is None
        assert r["identity"]["name"] is not None
