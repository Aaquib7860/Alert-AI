"""Phase 5 -- Validation split strategies.

Master plan section 10: "Never perform random row-level splits on repeated
customers or transactions." Two scenarios, reported separately (master
plan Experiment E): unseen customers (group split) and repeat customers
(time-forward split, same customer allowed in both train and eval, but
eval rows are strictly later in time).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit


def group_split_by_customer(
    df: pd.DataFrame,
    customer_col: str = "customer_id",
    test_size: float = 0.25,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    """Rows with a missing customer_id are excluded from both splits --
    they cannot be group-assigned without an identity, and silently
    dumping them into one side would bias that side's population.
    Returns (train_idx, test_idx) as positional (iloc-style) indices.
    """
    valid_mask = df[customer_col].notna().values
    valid_positions = np.where(valid_mask)[0]

    groups = df[customer_col].values[valid_positions]
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_rel, test_rel = next(splitter.split(valid_positions, groups=groups))

    train_idx = valid_positions[train_rel]
    test_idx = valid_positions[test_rel]

    # verify no customer appears on both sides -- the property this split
    # exists to guarantee, checked rather than assumed
    train_customers = set(df[customer_col].values[train_idx])
    test_customers = set(df[customer_col].values[test_idx])
    overlap = train_customers & test_customers
    assert not overlap, f"Group split leaked {len(overlap)} customer(s) across train/test"

    return train_idx, test_idx


def time_forward_split(
    df: pd.DataFrame,
    timestamp_col: str,
    test_frac: float = 0.25,
) -> tuple[np.ndarray, np.ndarray]:
    """Chronological split: earliest (1 - test_frac) rows train, latest
    test_frac rows test. Same customer CAN appear on both sides (that's
    the point -- this scenario represents realistic ongoing monitoring of
    an existing customer), but every test-side row is strictly later in
    time than every train-side row. Rows with a missing timestamp are
    excluded from both splits (cannot be chronologically placed).
    Returns (train_idx, test_idx) as positional indices.
    """
    valid_mask = df[timestamp_col].notna().values
    valid_positions = np.where(valid_mask)[0]
    valid_timestamps = df[timestamp_col].values[valid_positions]

    order = np.argsort(valid_timestamps, kind="mergesort")
    ordered_positions = valid_positions[order]

    cutoff = int(len(ordered_positions) * (1 - test_frac))
    train_idx = ordered_positions[:cutoff]
    test_idx = ordered_positions[cutoff:]

    if len(train_idx) and len(test_idx):
        max_train_ts = df[timestamp_col].values[train_idx].max()
        min_test_ts = df[timestamp_col].values[test_idx].min()
        assert max_train_ts <= min_test_ts, "Time-forward split is not chronologically ordered"

    return train_idx, test_idx
