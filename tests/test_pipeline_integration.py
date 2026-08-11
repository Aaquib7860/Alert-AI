from pathlib import Path

import pandas as pd
import pytest

from pipelines.ingestion.load_alerts import get_raw_data_path
from pipelines.normalization.pipeline import normalize_sheet, run_phase2_pipeline
from pipelines.validation.schema_registry import SCHEMA_REGISTRY
from pipelines.validation.temporal import POST_SCORING_COLUMNS


def _valid_df_for(sheet_name: str, n: int = 3) -> pd.DataFrame:
    schema = SCHEMA_REGISTRY[sheet_name]
    data = {}
    for col in schema.columns:
        data[col.name] = [None] * n
    return pd.DataFrame(data)


@pytest.mark.parametrize("sheet_name", list(SCHEMA_REGISTRY.keys()))
def test_normalize_sheet_runs_on_all_null_input(sheet_name):
    df = _valid_df_for(sheet_name)
    out = normalize_sheet(df, sheet_name)
    assert len(out) == len(df)
    assert "customer_id" in out.columns
    assert "record_id" in out.columns


@pytest.mark.parametrize("sheet_name", list(SCHEMA_REGISTRY.keys()))
def test_normalize_sheet_never_derives_a_feature_from_a_leakage_column(sheet_name):
    """Master plan section 19: 'Tests proving leakage fields cannot enter
    live inference.' No leakage-typed schema column should get a
    (Normalized)/(Parsed) companion column -- that would just be a leakage
    feature with a different name.
    """
    df = _valid_df_for(sheet_name)
    out = normalize_sheet(df, sheet_name)

    leakage_cols = SCHEMA_REGISTRY[sheet_name].leakage_columns
    for col in leakage_cols:
        assert f"{col} (Normalized)" not in out.columns
        assert f"{col} (Parsed)" not in out.columns
        # the leakage registry (Phase 1) and this schema registry must agree
        assert col in POST_SCORING_COLUMNS[sheet_name]


@pytest.mark.parametrize("sheet_name", list(SCHEMA_REGISTRY.keys()))
def test_normalize_sheet_preserves_raw_columns_untouched_in_value(sheet_name):
    """Raw source columns must survive normalization unmodified in *value*
    (companion columns are added, not substitutions) for every column that
    isn't one of the small set of known heterogeneous-type fields requiring
    a storage-safety cast (see pipeline.HETEROGENEOUS_DATE_COLUMNS)."""
    df = _valid_df_for(sheet_name, n=1)
    df.iloc[0] = "SAMPLE VALUE"
    out = normalize_sheet(df, sheet_name)
    for col in df.columns:
        assert out[col].iloc[0] == "SAMPLE VALUE"


RAW_DATA_AVAILABLE = get_raw_data_path().exists()


@pytest.mark.skipif(not RAW_DATA_AVAILABLE, reason="requires local Alerts_Samples.xlsx (gitignored, not present in CI)")
def test_full_pipeline_runs_on_real_data(tmp_path):
    normalized, report = run_phase2_pipeline(Path.cwd(), persist=False)
    assert report["overall_status"] == "PASS"
    assert set(normalized.keys()) == set(SCHEMA_REGISTRY.keys())
    for sheet_name, df in normalized.items():
        assert len(df) == report["sheets"][sheet_name]["raw_rows"]


@pytest.mark.skipif(not RAW_DATA_AVAILABLE, reason="requires local Alerts_Samples.xlsx (gitignored, not present in CI)")
def test_pipeline_persist_produces_parquet_and_manifest(tmp_path):
    normalized, report = run_phase2_pipeline(tmp_path, persist=True)
    version = report["dataset_version"]
    out_dir = tmp_path / "data" / "normalized" / version
    assert out_dir.exists()
    for sheet_name in SCHEMA_REGISTRY:
        assert (out_dir / f"{sheet_name}.parquet").exists()
    assert (out_dir / "manifest.json").exists()

    # round-trip check: parquet reload matches row count of the in-memory result
    reloaded = pd.read_parquet(out_dir / "CustomerViolation.parquet")
    assert len(reloaded) == len(normalized["CustomerViolation"])


@pytest.mark.skipif(not RAW_DATA_AVAILABLE, reason="requires local Alerts_Samples.xlsx (gitignored, not present in CI)")
def test_dataset_version_is_deterministic_for_same_source_file():
    _, report1 = run_phase2_pipeline(Path.cwd(), persist=False)
    _, report2 = run_phase2_pipeline(Path.cwd(), persist=False)
    assert report1["dataset_version"] == report2["dataset_version"]
