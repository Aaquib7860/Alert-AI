import numpy as np
import pandas as pd

from pipelines.evaluation.validation_manifest import (
    build_validation_manifest,
    load_validation_manifest,
    save_validation_manifest,
    verify_no_customer_leakage,
)


def _sample_df():
    return pd.DataFrame(
        {
            "record_id": [f"r{i}" for i in range(6)],
            "customer_id": ["c1", "c1", "c2", "c2", "c3", "c3"],
        }
    )


def test_manifest_captures_record_and_customer_ids():
    df = _sample_df()
    train_idx = np.array([0, 1, 2, 3])
    test_idx = np.array([4, 5])
    manifest = build_validation_manifest(df, train_idx, test_idx, "test_scenario", "v1", "fv1")

    assert manifest.train_record_ids == ["r0", "r1", "r2", "r3"]
    assert manifest.test_record_ids == ["r4", "r5"]
    assert set(manifest.train_customer_ids) == {"c1", "c2"}
    assert set(manifest.test_customer_ids) == {"c3"}


def test_manifest_no_leakage_detected_for_group_split():
    df = _sample_df()
    train_idx = np.array([0, 1, 2, 3])  # c1, c2
    test_idx = np.array([4, 5])  # c3
    manifest = build_validation_manifest(df, train_idx, test_idx, "group", "v1", "fv1")
    assert verify_no_customer_leakage(manifest)


def test_manifest_detects_leakage_for_time_forward_split():
    df = _sample_df()
    train_idx = np.array([0, 1, 2])  # c1, c2
    test_idx = np.array([3, 4, 5])  # c2, c3 -- c2 appears on both sides
    manifest = build_validation_manifest(df, train_idx, test_idx, "time_forward", "v1", "fv1")
    assert not verify_no_customer_leakage(manifest)


def test_manifest_save_and_load_round_trip(tmp_path):
    df = _sample_df()
    manifest = build_validation_manifest(df, np.array([0, 1]), np.array([2, 3]), "scn", "v1", "fv1")
    path = save_validation_manifest(manifest, tmp_path)
    assert path.exists()

    loaded = load_validation_manifest(path)
    assert loaded["scenario_name"] == "scn"
    assert loaded["train_record_ids"] == ["r0", "r1"]
    assert loaded["n_train"] == 2
    assert loaded["n_test"] == 2


def test_manifest_as_dict_reports_customer_overlap_count():
    df = _sample_df()
    train_idx = np.array([0, 1, 2])  # c1, c2
    test_idx = np.array([3, 4, 5])  # c2, c3
    manifest = build_validation_manifest(df, train_idx, test_idx, "scn", "v1", "fv1")
    d = manifest.as_dict()
    assert d["customer_overlap_count"] == 1
