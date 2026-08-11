import numpy as np
import pandas as pd
import pytest

from pipelines.entity.validation_splits import group_split_by_customer, time_forward_split


def test_group_split_no_customer_overlap():
    df = pd.DataFrame(
        {
            "customer_id": [f"c{i % 10}" for i in range(100)],
        }
    )
    train_idx, test_idx = group_split_by_customer(df, test_size=0.3)
    train_customers = set(df["customer_id"].iloc[train_idx])
    test_customers = set(df["customer_id"].iloc[test_idx])
    assert not (train_customers & test_customers)
    assert len(train_idx) + len(test_idx) == len(df)


def test_group_split_excludes_missing_customer_id():
    df = pd.DataFrame({"customer_id": ["c1", "c1", None, "c2", "c2", "c3", "c3", "c4"]})
    train_idx, test_idx = group_split_by_customer(df, test_size=0.3)
    all_idx = set(train_idx) | set(test_idx)
    assert 2 not in all_idx  # the None row (positional index 2) excluded


def test_time_forward_split_is_chronological():
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=100, freq="D")})
    train_idx, test_idx = time_forward_split(df, "ts", test_frac=0.2)
    assert df["ts"].iloc[train_idx].max() <= df["ts"].iloc[test_idx].min()
    assert len(train_idx) + len(test_idx) == len(df)


def test_time_forward_split_allows_same_customer_both_sides():
    df = pd.DataFrame(
        {
            "customer_id": ["c1"] * 50 + ["c2"] * 50,
            "ts": pd.date_range("2024-01-01", periods=100, freq="D"),
        }
    )
    train_idx, test_idx = time_forward_split(df, "ts", test_frac=0.3)
    train_customers = set(df["customer_id"].iloc[train_idx])
    test_customers = set(df["customer_id"].iloc[test_idx])
    assert train_customers & test_customers  # overlap IS expected here


def test_time_forward_split_excludes_missing_timestamps():
    df = pd.DataFrame({"ts": [pd.Timestamp("2024-01-01"), pd.NaT, pd.Timestamp("2024-01-02")]})
    train_idx, test_idx = time_forward_split(df, "ts", test_frac=0.5)
    all_idx = set(train_idx) | set(test_idx)
    assert 1 not in all_idx
