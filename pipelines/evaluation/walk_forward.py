"""Phase 7 -- Chronological walk-forward replay.

Feasibility report Experiment E / master plan section 10 (chronological
holdout, "Production replay"). Phase 5/6 each used a single time-forward
split (earliest 75% train, latest 25% test). This module goes further:
it splits the full historical period into ordered folds and, for each fold
after the first, trains only on everything strictly before that fold and
scores the fold itself -- simulating "if this model had been deployed and
retrained periodically, how would its score distribution have behaved
period over period." This is still not a claim of supervised accuracy
(master plan section 11) -- it evaluates distribution stability, not
correctness against a label.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


@dataclass
class WalkForwardFold:
    fold_index: int
    train_idx: np.ndarray
    score_idx: np.ndarray


def chronological_folds(
    df: pd.DataFrame, timestamp_col: str, n_folds: int = 4,
) -> list[WalkForwardFold]:
    """Splits rows with a valid timestamp into `n_folds` equal-sized,
    chronologically ordered chunks. Fold 0 has no prior data (its
    `train_idx` is empty) so it is scoring-only and typically skipped by
    callers; folds 1..n-1 each train on every fold strictly before them.
    Rows with a missing timestamp are excluded entirely (cannot be
    chronologically placed).
    """
    valid_mask = df[timestamp_col].notna().values
    valid_positions = np.where(valid_mask)[0]
    valid_timestamps = df[timestamp_col].values[valid_positions]

    order = np.argsort(valid_timestamps, kind="mergesort")
    ordered_positions = valid_positions[order]

    chunks = np.array_split(ordered_positions, n_folds)
    folds = []
    for i, chunk in enumerate(chunks):
        train_idx = np.concatenate(chunks[:i]) if i > 0 else np.array([], dtype=int)
        folds.append(WalkForwardFold(fold_index=i, train_idx=train_idx, score_idx=chunk))
    return folds


def run_walk_forward(
    df: pd.DataFrame,
    timestamp_col: str,
    fit_transform_fn,
    fit_model_fn,
    score_model_fn,
    n_folds: int = 4,
    min_train_size: int = 20,
) -> list[dict]:
    """Generic walk-forward runner.

    `fit_transform_fn(train_df, test_df) -> (X_train, X_test)` builds
    whatever representation the caller needs (already-fit-on-train-only).
    `fit_model_fn(X_train)` and `score_model_fn(model, X_test)` wrap a
    specific anomaly model. Folds with fewer than `min_train_size` training
    rows are skipped (score_samples on a near-empty train set is not
    meaningful) and reported as such, not silently omitted.
    """
    folds = chronological_folds(df, timestamp_col, n_folds=n_folds)
    results = []

    for fold in folds:
        if len(fold.train_idx) < min_train_size:
            results.append({
                "fold_index": fold.fold_index,
                "skipped": True,
                "reason": f"train size {len(fold.train_idx)} < min_train_size {min_train_size}",
                "n_train": len(fold.train_idx),
                "n_score": len(fold.score_idx),
            })
            continue

        train_df = df.iloc[fold.train_idx].reset_index(drop=True)
        score_df = df.iloc[fold.score_idx].reset_index(drop=True)

        X_train, X_score = fit_transform_fn(train_df, score_df)
        model = fit_model_fn(X_train)
        scores = score_model_fn(model, X_score)

        results.append({
            "fold_index": fold.fold_index,
            "skipped": False,
            "n_train": len(fold.train_idx),
            "n_score": len(fold.score_idx),
            "score_mean": float(np.mean(scores)),
            "score_std": float(np.std(scores)),
            "score_min": float(np.min(scores)),
            "score_max": float(np.max(scores)),
        })

    return results
