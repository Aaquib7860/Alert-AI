"""Phase 5 -- Unsupervised evaluation framework.

Master plan section 11: "Because the current dataset lacks trustworthy
ground truth, the agent must not report generic classification accuracy."
Every function here evaluates *ranking usefulness*, *stability*, and
*distribution sanity* -- never a label-based metric. The one function that
touches Status/UPS (`exploratory_status_comparison`) is explicitly labeled
exploratory-only and must never feed champion selection.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import spearmanr


def score_distribution_summary(scores: np.ndarray) -> dict:
    """A degenerate model (near-constant score for every row) is useless
    for ranking regardless of anything else -- this is the first, cheapest
    check before looking at anything more elaborate.
    """
    return {
        "mean": float(np.mean(scores)),
        "std": float(np.std(scores)),
        "min": float(np.min(scores)),
        "max": float(np.max(scores)),
        "p50": float(np.percentile(scores, 50)),
        "p90": float(np.percentile(scores, 90)),
        "p99": float(np.percentile(scores, 99)),
        "is_degenerate": bool(np.std(scores) < 1e-8),
    }


def top_n_novelty_table(
    df: pd.DataFrame, scores: np.ndarray, display_cols: list[str], n: int = 15,
) -> pd.DataFrame:
    """Highest- and lowest-novelty rows, for manual inspection (master plan:
    "Inspect the highest-novelty and lowest-novelty historical alerts
    manually"). `display_cols` should be pre-decision context only -- this
    function does not enforce that; the caller decides what's safe to show.
    """
    out = df[display_cols].copy()
    out["novelty_score"] = scores
    out["novelty_rank"] = pd.Series(scores, index=df.index).rank(ascending=False, method="first")
    top = out.sort_values("novelty_score", ascending=False).head(n)
    bottom = out.sort_values("novelty_score", ascending=True).head(n)
    return pd.concat([top.assign(group="top"), bottom.assign(group="bottom")])


def novelty_by_segment(df: pd.DataFrame, scores: np.ndarray, segment_col: str) -> pd.DataFrame:
    """Mean/std novelty per segment (screening list, alert type, branch,
    alert_source_sheet). Master plan: "Novelty concentration by rule,
    watchlist, branch and customer segment."
    """
    tmp = pd.DataFrame({segment_col: df[segment_col].values, "score": scores})
    return tmp.groupby(segment_col)["score"].agg(["mean", "std", "count"]).sort_values(
        "mean", ascending=False
    )


def score_stability_across_time(
    df: pd.DataFrame, scores: np.ndarray, timestamp_col: str, n_bins: int = 4,
) -> pd.DataFrame:
    """Master plan: "Stability of anomaly rankings across time windows."
    A model whose score distribution drifts wildly period-to-period (with
    no external population change to explain it) is less trustworthy than
    one that stays roughly consistent.
    """
    valid = df[timestamp_col].notna()
    tmp = pd.DataFrame({"ts": df.loc[valid, timestamp_col].values, "score": scores[valid.values]})
    tmp["period"] = pd.qcut(tmp["ts"].rank(method="first"), q=n_bins, labels=False, duplicates="drop")
    return tmp.groupby("period")["score"].agg(["mean", "std", "count"])


def customer_history_consistency(
    df: pd.DataFrame, scores: np.ndarray, customer_col: str = "customer_id", min_alerts: int = 3,
) -> dict:
    """Master plan: "Cluster/representation consistency for repeated
    customers." For customers with several alerts, how much does their own
    score vary? Very high variance for a customer whose profile barely
    changes suggests the representation is noisy, not that the customer is
    genuinely erratic -- worth knowing before trusting the model's ranking.
    """
    tmp = pd.DataFrame({customer_col: df[customer_col].values, "score": scores})
    tmp = tmp.dropna(subset=[customer_col])
    counts = tmp.groupby(customer_col).size()
    repeat_customers = counts[counts >= min_alerts].index
    if len(repeat_customers) == 0:
        return {"n_repeat_customers_evaluated": 0, "median_within_customer_std": None}

    within_std = tmp[tmp[customer_col].isin(repeat_customers)].groupby(customer_col)["score"].std()
    return {
        "n_repeat_customers_evaluated": int(len(repeat_customers)),
        "median_within_customer_std": float(within_std.median()),
        "overall_score_std": float(np.std(scores)),
    }


def ranking_stability_between_models(scores_a: np.ndarray, scores_b: np.ndarray) -> float:
    """Spearman correlation between two independently-fit models' scores on
    the same rows (e.g. two different random train subsamples). Used as
    part of champion selection: a model whose ranking changes wildly
    depending on which random subset of data it happened to be fit on is
    less trustworthy than one that converges to a similar ranking.
    """
    corr, _ = spearmanr(scores_a, scores_b)
    return float(corr)


def exploratory_status_comparison(
    df: pd.DataFrame, scores: np.ndarray, status_col: str,
) -> pd.DataFrame:
    """EXPLORATORY ONLY -- master plan section 5 ("It should not claim that
    Released means confirmed false positive... UPS means confirmed true
    match") and section 11 ("only as an exploratory analysis, not as
    ground-truth accuracy"). This function must NEVER be used to select a
    champion model or compute a claimed accuracy figure. It exists purely
    to sanity-check whether novelty correlates at all with historical
    operational handling -- a correlation here is neither required nor
    sufficient for the model to be considered good, because Status itself
    is not trustworthy ground truth (Phase 1 finding: a majority of UPS
    rows in two of three sheets literally contain "false positive" in the
    review comment).
    """
    tmp = pd.DataFrame({status_col: df[status_col].values, "score": scores})
    return tmp.groupby(status_col)["score"].agg(["mean", "std", "count"])


@dataclass
class ExperimentResult:
    experiment_id: str
    model_name: str
    representation: str
    validation_scenario: str
    distribution: dict
    stability_spearman: float | None
    history_consistency: dict
    fit_seconds: float
    score_seconds: float

    def as_dict(self) -> dict:
        return {
            "experiment_id": self.experiment_id,
            "model_name": self.model_name,
            "representation": self.representation,
            "validation_scenario": self.validation_scenario,
            "distribution": self.distribution,
            "stability_spearman": self.stability_spearman,
            "history_consistency": self.history_consistency,
            "fit_seconds": self.fit_seconds,
            "score_seconds": self.score_seconds,
        }


def select_champion(results: list[ExperimentResult]) -> tuple[ExperimentResult, dict]:
    """Documented, reproducible champion-selection rubric -- NOT accuracy,
    NOT any comparison to Status/UPS/Released (master plan section 11).

    Selection order:
      1. Drop degenerate models (near-constant score -- useless for ranking).
      2. Among the rest, prefer higher bootstrap ranking stability
         (Spearman correlation between two independently-fit models on the
         same holdout -- master plan section 11: "Stability of anomaly
         rankings across time windows" generalized to fit-to-fit stability).
      3. Tie-break: lower within-repeat-customer score variance relative to
         overall score variance (a consistent representation for a customer
         whose profile barely changes, master plan: "Cluster/representation
         consistency for repeated customers").
      4. Tie-break: lower combined fit+score time (master plan section 4:
         "cost-sensitive").

    Returns (champion, rubric_detail) where rubric_detail explains the
    reasoning for audit -- champion selection must be evidence-based and
    reproducible, not asserted.
    """
    candidates = [r for r in results if not r.distribution["is_degenerate"]]
    dropped_degenerate = [r.experiment_id for r in results if r.distribution["is_degenerate"]]

    if not candidates:
        raise ValueError("Every candidate experiment produced a degenerate score distribution")

    def _consistency_ratio(r: ExperimentResult) -> float:
        hc = r.history_consistency
        if hc.get("median_within_customer_std") is None or not hc.get("overall_score_std"):
            return float("inf")  # no repeat-customer evidence -- worst, not best, by default
        return hc["median_within_customer_std"] / hc["overall_score_std"]

    def _sort_key(r: ExperimentResult):
        stability = r.stability_spearman if r.stability_spearman is not None else -1.0
        return (-stability, _consistency_ratio(r), r.fit_seconds + r.score_seconds)

    ranked = sorted(candidates, key=_sort_key)
    champion = ranked[0]

    rubric_detail = {
        "dropped_degenerate_experiment_ids": dropped_degenerate,
        "ranked_experiment_ids_best_to_worst": [r.experiment_id for r in ranked],
        "champion_experiment_id": champion.experiment_id,
        "champion_stability_spearman": champion.stability_spearman,
        "champion_consistency_ratio": _consistency_ratio(champion),
        "selection_criteria_order": [
            "not degenerate", "highest bootstrap ranking stability (Spearman)",
            "lowest within-customer/overall score-std ratio", "lowest fit+score time",
        ],
    }
    return champion, rubric_detail
