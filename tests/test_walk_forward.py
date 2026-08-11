import numpy as np
import pandas as pd
import pytest

from pipelines.evaluation.walk_forward import chronological_folds, run_walk_forward


def test_chronological_folds_first_fold_has_no_train_data():
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=40, freq="D")})
    folds = chronological_folds(df, "ts", n_folds=4)
    assert len(folds) == 4
    assert len(folds[0].train_idx) == 0
    assert len(folds[0].score_idx) == 10


def test_chronological_folds_train_grows_each_fold():
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=40, freq="D")})
    folds = chronological_folds(df, "ts", n_folds=4)
    train_sizes = [len(f.train_idx) for f in folds]
    assert train_sizes == sorted(train_sizes)  # non-decreasing


def test_chronological_folds_train_always_strictly_before_score_fold():
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=40, freq="D")})
    folds = chronological_folds(df, "ts", n_folds=4)
    for fold in folds[1:]:
        max_train_ts = df["ts"].iloc[fold.train_idx].max()
        min_score_ts = df["ts"].iloc[fold.score_idx].min()
        assert max_train_ts < min_score_ts


def test_chronological_folds_excludes_missing_timestamps():
    df = pd.DataFrame({"ts": [pd.Timestamp("2024-01-01"), pd.NaT, pd.Timestamp("2024-01-02")]})
    folds = chronological_folds(df, "ts", n_folds=2)
    total_rows = sum(len(f.score_idx) for f in folds)
    assert total_rows == 2  # the NaT row is excluded


def test_run_walk_forward_skips_folds_below_min_train_size():
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=20, freq="D")})

    def fit_transform_fn(train_df, score_df):
        return np.zeros((len(train_df), 2)), np.zeros((len(score_df), 2))

    def fit_model_fn(X_train):
        return "dummy_model"

    def score_model_fn(model, X_score):
        return np.ones(X_score.shape[0])

    results = run_walk_forward(
        df, "ts", fit_transform_fn, fit_model_fn, score_model_fn, n_folds=4, min_train_size=1000,
    )
    assert all(r["skipped"] for r in results)  # every fold's train size is well under 1000


def test_run_walk_forward_produces_score_stats_for_valid_folds():
    df = pd.DataFrame({"ts": pd.date_range("2024-01-01", periods=40, freq="D")})

    def fit_transform_fn(train_df, score_df):
        return np.zeros((len(train_df), 2)), np.arange(len(score_df)).reshape(-1, 1).astype(float)

    def fit_model_fn(X_train):
        return "dummy_model"

    def score_model_fn(model, X_score):
        return X_score.flatten()

    results = run_walk_forward(
        df, "ts", fit_transform_fn, fit_model_fn, score_model_fn, n_folds=4, min_train_size=1,
    )
    non_skipped = [r for r in results if not r["skipped"]]
    assert len(non_skipped) == 3  # folds 1,2,3 have train data; fold 0 doesn't
    for r in non_skipped:
        assert "score_mean" in r and "score_std" in r
