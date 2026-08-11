"""Phase 7 -- Validation manifests.

Master plan section 10: "The agent must generate a validation manifest for
every run showing exactly which rows/groups belong to training, validation
and test populations." This is an audit artifact, not PII -- it records
`record_id` (an opaque sha1 hash, see pipelines/ingestion/identifiers.py),
never a name/DOB/UIN, so it is safe to persist and commit.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd


@dataclass
class ValidationManifest:
    scenario_name: str
    dataset_version: str
    feature_version: str
    train_record_ids: list[str]
    test_record_ids: list[str]
    train_customer_ids: list[str] = field(default_factory=list)
    test_customer_ids: list[str] = field(default_factory=list)
    generated_at: str = ""

    def as_dict(self) -> dict:
        return {
            "scenario_name": self.scenario_name,
            "dataset_version": self.dataset_version,
            "feature_version": self.feature_version,
            "generated_at": self.generated_at,
            "n_train": len(self.train_record_ids),
            "n_test": len(self.test_record_ids),
            "train_record_ids": self.train_record_ids,
            "test_record_ids": self.test_record_ids,
            "n_train_unique_customers": len(set(self.train_customer_ids)),
            "n_test_unique_customers": len(set(self.test_customer_ids)),
            "customer_overlap_count": len(
                set(self.train_customer_ids) & set(self.test_customer_ids)
            ),
        }


def build_validation_manifest(
    df: pd.DataFrame,
    train_idx: np.ndarray,
    test_idx: np.ndarray,
    scenario_name: str,
    dataset_version: str,
    feature_version: str,
    record_id_col: str = "record_id",
    customer_id_col: str = "customer_id",
) -> ValidationManifest:
    return ValidationManifest(
        scenario_name=scenario_name,
        dataset_version=dataset_version,
        feature_version=feature_version,
        train_record_ids=df[record_id_col].iloc[train_idx].tolist(),
        test_record_ids=df[record_id_col].iloc[test_idx].tolist(),
        train_customer_ids=df[customer_id_col].iloc[train_idx].dropna().tolist(),
        test_customer_ids=df[customer_id_col].iloc[test_idx].dropna().tolist(),
        generated_at=datetime.now(timezone.utc).isoformat(),
    )


def save_validation_manifest(manifest: ValidationManifest, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{manifest.scenario_name}_manifest.json"
    with open(path, "w") as f:
        json.dump(manifest.as_dict(), f, indent=2)
    return path


def load_validation_manifest(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def verify_no_customer_leakage(manifest: ValidationManifest) -> bool:
    """True if train and test customer populations are disjoint. Expected
    True for group-split (unseen customer) scenarios, expected False for
    time-forward (repeat customer) scenarios -- callers should check the
    scenario type before treating a False result as a problem.
    """
    return not (set(manifest.train_customer_ids) & set(manifest.test_customer_ids))
