import pandas as pd
import pytest

from pipelines.entity.combined_dataset import SHARED_COLUMNS, build_combined_entity_dataset


def _minimal_shared_df():
    return pd.DataFrame({c: [None] for c in SHARED_COLUMNS})


def test_combined_dataset_row_count_is_sum_of_inputs():
    cv = pd.concat([_minimal_shared_df()] * 3, ignore_index=True)
    tnv = pd.concat([_minimal_shared_df()] * 2, ignore_index=True)
    combined = build_combined_entity_dataset(cv, tnv)
    assert len(combined) == 5


def test_combined_dataset_tags_source_sheet():
    cv = _minimal_shared_df()
    tnv = _minimal_shared_df()
    combined = build_combined_entity_dataset(cv, tnv)
    assert combined["alert_source_sheet"].tolist() == ["CustomerViolation", "TransactionNameViolation"]


def test_combined_dataset_missing_shared_column_raises():
    cv = _minimal_shared_df().drop(columns=["customer_id"])
    tnv = _minimal_shared_df()
    with pytest.raises(ValueError):
        build_combined_entity_dataset(cv, tnv)
