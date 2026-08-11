"""Phase 8 -- Model registry access.

Reads `models/registry/active_models.json` (built by
scripts/build_scoring_registry.py) to find which model version is
currently active per alert type, then loads the corresponding champion
artifact (joblib -- fitted model + SVD + feature artifacts) and scoring
registry (JSON -- reference score distribution + per-customer baselines).

Master plan section 15: no hardcoded paths -- the registry root is
configurable via `MODEL_REGISTRY_PATH` (see .env.example), defaulting to
the repo-relative `models/` used throughout this project.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib

REPO_ROOT = Path(__file__).resolve().parents[2]

ALERT_TYPE_TO_SHEET = {
    "customer_name": "CustomerViolation",
    "transaction_name": "TransactionNameViolation",
    "transaction_rule": "Rule",
}


class ModelNotActiveError(RuntimeError):
    pass


def get_models_root() -> Path:
    configured = os.environ.get("MODEL_REGISTRY_PATH")
    if configured:
        path = Path(configured)
        return path if path.is_absolute() else REPO_ROOT / configured
    return REPO_ROOT / "models"


@lru_cache(maxsize=1)
def _load_active_models_pointer() -> dict:
    path = get_models_root() / "registry" / "active_models.json"
    if not path.exists():
        raise ModelNotActiveError(
            f"No active-models pointer at {path}. Run "
            "scripts/build_scoring_registry.py first."
        )
    with open(path) as f:
        return json.load(f)


@dataclass
class LoadedModel:
    alert_type: str
    sheet_name: str
    model: object
    model_kind: str
    representation: str
    svd: object | None
    feature_artifacts: object
    train_score_distribution: list[float]
    customer_baselines: dict
    model_version: str
    feature_version: str
    schema_version: str


def _artifacts_key_for(alert_type: str) -> str:
    return "entity_feature_artifacts" if alert_type in ("customer_name", "transaction_name") else "transaction_feature_artifacts"


@lru_cache(maxsize=8)
def load_active_model(alert_type: str) -> LoadedModel:
    if alert_type not in ALERT_TYPE_TO_SHEET:
        raise ValueError(
            f"Unknown alert_type {alert_type!r}. Must be one of {list(ALERT_TYPE_TO_SHEET)}"
        )

    pointer = _load_active_models_pointer()
    if alert_type not in pointer:
        raise ModelNotActiveError(f"No active model registered for alert_type={alert_type!r}")

    entry = pointer[alert_type]
    registry_file = Path(entry["registry_file"])
    registry_path = registry_file if registry_file.is_absolute() else REPO_ROOT / registry_file
    with open(registry_path) as f:
        registry = json.load(f)

    champion_dir = registry_path.parent
    champion_manifest = json.load(open(champion_dir / "champion_manifest.json"))
    payload = joblib.load(champion_dir / champion_manifest["artifact_file"])

    return LoadedModel(
        alert_type=alert_type,
        sheet_name=ALERT_TYPE_TO_SHEET[alert_type],
        model=payload["model"],
        model_kind=payload["model_kind"],
        representation=payload["representation"],
        svd=payload.get("svd"),
        feature_artifacts=payload[_artifacts_key_for(alert_type)],
        train_score_distribution=registry["train_score_distribution"],
        customer_baselines=registry["customer_baselines"],
        model_version=registry["model_version"],
        feature_version=registry["feature_version"],
        schema_version=registry["schema_version"],
    )


def clear_model_cache() -> None:
    """For tests / after re-running build_scoring_registry.py without
    restarting the process.
    """
    _load_active_models_pointer.cache_clear()
    load_active_model.cache_clear()
