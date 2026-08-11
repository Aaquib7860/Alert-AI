import pytest

from app.services.model_registry import (
    ALERT_TYPE_TO_SHEET,
    ModelNotActiveError,
    clear_model_cache,
    get_models_root,
    load_active_model,
)

REGISTRY_AVAILABLE = (get_models_root() / "entity" / "scoring_registry.json").exists() and (
    get_models_root() / "transaction" / "scoring_registry.json"
).exists()


def test_get_models_root_defaults_to_repo_models_dir():
    root = get_models_root()
    assert root.name == "models"


def test_load_active_model_unknown_alert_type_raises():
    with pytest.raises(ValueError):
        load_active_model("not_a_real_type")


@pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="requires scripts/build_scoring_registry.py to have been run locally")
@pytest.mark.parametrize("alert_type", list(ALERT_TYPE_TO_SHEET.keys()))
def test_load_active_model_returns_populated_loaded_model(alert_type):
    clear_model_cache()
    loaded = load_active_model(alert_type)
    assert loaded.alert_type == alert_type
    assert loaded.model is not None
    assert loaded.model_kind
    assert loaded.model_version
    assert loaded.feature_version
    assert loaded.schema_version
    assert len(loaded.train_score_distribution) > 0
    assert len(loaded.customer_baselines) > 0


@pytest.mark.skipif(not REGISTRY_AVAILABLE, reason="requires scripts/build_scoring_registry.py to have been run locally")
def test_load_active_model_is_cached():
    clear_model_cache()
    loaded1 = load_active_model("customer_name")
    loaded2 = load_active_model("customer_name")
    assert loaded1 is loaded2  # same cached object, not re-loaded from disk
