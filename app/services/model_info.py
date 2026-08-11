"""Phase 8 -- Model info lookup for GET /api/v1/models/* endpoints."""
from __future__ import annotations

from app.services.model_registry import ALERT_TYPE_TO_SHEET, ModelNotActiveError, load_active_model


def get_model_info(alert_type: str) -> dict:
    loaded = load_active_model(alert_type)
    return {
        "alert_type": alert_type,
        "model_version": loaded.model_version,
        "feature_version": loaded.feature_version,
        "schema_version": loaded.schema_version,
        "model_kind": loaded.model_kind,
        "representation": loaded.representation,
        "dataset_version": loaded.model_version.split("@", 1)[-1],
        "n_reference_scores": len(loaded.train_score_distribution),
        "n_customer_baselines": len(loaded.customer_baselines),
    }


def get_all_active_model_info() -> list[dict]:
    infos = []
    for alert_type in ALERT_TYPE_TO_SHEET:
        try:
            infos.append(get_model_info(alert_type))
        except ModelNotActiveError:
            continue
    return infos


def find_model_info_by_version(model_version: str) -> dict | None:
    for info in get_all_active_model_info():
        if info["model_version"] == model_version:
            return info
    return None


def models_health() -> dict[str, bool]:
    health = {}
    for alert_type in ALERT_TYPE_TO_SHEET:
        try:
            load_active_model(alert_type)
            health[alert_type] = True
        except ModelNotActiveError:
            health[alert_type] = False
    return health
