"""Phase 8 -- Build the scoring registry.

Master plan section 16 (Model Lifecycle): ... -> COMPARE CHAMPIONS -> MODEL
CARD -> REGISTER -> SHADOW MODE -> ... This script performs the "Register"
step: takes the Phase 5/6 champions (already selected on documented
evidence) and derives everything the live scoring API needs but that a
raw fitted model doesn't carry on its own --

  - a reference training-score distribution, so a raw anomaly score can be
    reported as a percentile (bounded, interpretable) rather than a raw
    unbounded number whose scale drifts across retrains (Phase 7 finding)
  - per-customer score baselines (mean/std/n), so a "customer-specific"
    novelty figure is possible for repeat customers (master plan
    architecture diagram: "Global anomaly model" + "Customer-specific
    baseline" as parallel outputs)

Run this after any Phase 5/6/7 re-run to refresh the registry. Output is
gitignored (models/entity/*, models/transaction/*, models/registry/*) --
score arrays and per-customer stats are derived, non-reversible aggregates
(no raw PII), but kept local-only for consistency with every other model
artifact in this repo.
"""
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]

import sys
sys.path.insert(0, str(REPO_ROOT))

from pipelines.entity.anomaly_models import extract_name_representation_matrix, score_ocsvm
from pipelines.entity.combined_dataset import build_combined_entity_dataset
from pipelines.entity.validation_splits import group_split_by_customer
from pipelines.normalization.pipeline import run_phase2_pipeline
from pipelines.validation.schema_registry import SCHEMA_VERSION
from features.entity_features import transform_entity_features
from features.transaction_features import transform_transaction_features

SCORE_FN_BY_MODEL_KIND = {
    "ocsvm": score_ocsvm,
}


def _score_fn(model_kind: str):
    if model_kind not in SCORE_FN_BY_MODEL_KIND:
        raise ValueError(
            f"No score function wired up for model_kind={model_kind!r}. "
            "Add it to SCORE_FN_BY_MODEL_KIND if the champion changes."
        )
    return SCORE_FN_BY_MODEL_KIND[model_kind]


def _customer_baselines(scores: np.ndarray, customer_ids: list) -> dict:
    stats = {}
    by_customer: dict[str, list[float]] = {}
    for cid, score in zip(customer_ids, scores):
        if cid is None or (isinstance(cid, float) and np.isnan(cid)):
            continue
        by_customer.setdefault(cid, []).append(float(score))
    for cid, vals in by_customer.items():
        arr = np.array(vals)
        stats[cid] = {
            "mean": float(arr.mean()),
            "std": float(arr.std()) if len(arr) > 1 else 0.0,
            "n": len(arr),
        }
    return stats


def build_entity_registry() -> dict:
    normalized_sheets, phase2_report = run_phase2_pipeline(REPO_ROOT, persist=False)
    combined = build_combined_entity_dataset(
        normalized_sheets["CustomerViolation"], normalized_sheets["TransactionNameViolation"]
    )

    champion_manifest = json.load(open(REPO_ROOT / "models" / "entity" / "champion_manifest.json"))
    champion_path = REPO_ROOT / "models" / "entity" / champion_manifest["artifact_file"]
    payload = joblib.load(champion_path)

    train_idx, _ = group_split_by_customer(combined, test_size=0.25, random_state=42)
    train_df = combined.iloc[train_idx].reset_index(drop=True)

    artifacts = payload["entity_feature_artifacts"]
    train_matrix, _ = transform_entity_features(train_df, "CombinedEntity", artifacts)
    name_train = extract_name_representation_matrix(train_matrix, artifacts)
    X_train = payload["svd"].transform(name_train) if payload["svd"] is not None else name_train

    score_fn = _score_fn(payload["model_kind"])
    train_scores = score_fn(payload["model"], X_train)

    registry = {
        "alert_type_sheets": ["CustomerViolation", "TransactionNameViolation"],
        "model_kind": payload["model_kind"],
        "representation": payload["representation"],
        "experiment_id": payload["experiment_id"],
        "model_version": f"{payload['experiment_id']}@{phase2_report['dataset_version']}",
        "feature_version": artifacts.feature_version,
        "schema_version": SCHEMA_VERSION,
        "dataset_version": phase2_report["dataset_version"],
        "train_score_distribution": sorted(float(s) for s in train_scores),
        "customer_baselines": _customer_baselines(train_scores, train_df["customer_id"].tolist()),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = REPO_ROOT / "models" / "entity" / "scoring_registry.json"
    with open(out_path, "w") as f:
        json.dump(registry, f)
    print(f"Entity scoring registry written: {out_path} "
          f"({len(registry['train_score_distribution'])} reference scores, "
          f"{len(registry['customer_baselines'])} customer baselines)")
    return registry


def build_transaction_registry() -> dict:
    normalized_sheets, phase2_report = run_phase2_pipeline(REPO_ROOT, persist=False)
    rule_df = normalized_sheets["Rule"]

    champion_manifest = json.load(open(REPO_ROOT / "models" / "transaction" / "champion_manifest.json"))
    champion_path = REPO_ROOT / "models" / "transaction" / champion_manifest["artifact_file"]
    payload = joblib.load(champion_path)

    train_idx, _ = group_split_by_customer(rule_df, test_size=0.25, random_state=42)
    train_df = rule_df.iloc[train_idx].reset_index(drop=True)

    artifacts = payload["transaction_feature_artifacts"]
    train_matrix, _ = transform_transaction_features(train_df, artifacts)
    X_train = payload["svd"].transform(train_matrix) if payload["svd"] is not None else train_matrix

    score_fn = _score_fn(payload["model_kind"])
    train_scores = score_fn(payload["model"], X_train)

    registry = {
        "alert_type_sheets": ["Rule"],
        "model_kind": payload["model_kind"],
        "representation": payload["representation"],
        "experiment_id": payload["experiment_id"],
        "model_version": f"{payload['experiment_id']}@{phase2_report['dataset_version']}",
        "feature_version": artifacts.feature_version,
        "schema_version": SCHEMA_VERSION,
        "dataset_version": phase2_report["dataset_version"],
        "train_score_distribution": sorted(float(s) for s in train_scores),
        "customer_baselines": _customer_baselines(train_scores, train_df["customer_id"].tolist()),
        "built_at": datetime.now(timezone.utc).isoformat(),
    }

    out_path = REPO_ROOT / "models" / "transaction" / "scoring_registry.json"
    with open(out_path, "w") as f:
        json.dump(registry, f)
    print(f"Transaction scoring registry written: {out_path} "
          f"({len(registry['train_score_distribution'])} reference scores, "
          f"{len(registry['customer_baselines'])} customer baselines)")
    return registry


def build_active_models_pointer(entity_registry: dict, transaction_registry: dict) -> Path:
    """models/registry/active_models.json -- the file the live API reads to
    know which model version is currently active per alert type. Editing
    this file (not the code) is how a future promotion/rollback happens
    (master plan section 16: REGISTER -> SHADOW MODE -> ... -> PROMOTE OR
    REJECT), without redeploying application code.
    """
    pointer = {
        "customer_name": {
            "registry_file": "models/entity/scoring_registry.json",
            "model_version": entity_registry["model_version"],
        },
        "transaction_name": {
            "registry_file": "models/entity/scoring_registry.json",
            "model_version": entity_registry["model_version"],
        },
        "transaction_rule": {
            "registry_file": "models/transaction/scoring_registry.json",
            "model_version": transaction_registry["model_version"],
        },
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    out_dir = REPO_ROOT / "models" / "registry"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "active_models.json"
    with open(out_path, "w") as f:
        json.dump(pointer, f, indent=2)
    print(f"Active-models pointer written: {out_path}")
    return out_path


if __name__ == "__main__":
    entity_reg = build_entity_registry()
    transaction_reg = build_transaction_registry()
    build_active_models_pointer(entity_reg, transaction_reg)
