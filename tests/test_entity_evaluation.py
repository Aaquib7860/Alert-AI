import numpy as np
import pandas as pd
import pytest

from pipelines.entity.evaluation import (
    customer_history_consistency,
    exploratory_status_comparison,
    novelty_by_segment,
    ranking_stability_between_models,
    score_distribution_summary,
    score_stability_across_time,
    top_n_novelty_table,
)


def test_score_distribution_summary_detects_degenerate_scores():
    constant_scores = np.full(50, 0.5)
    summary = score_distribution_summary(constant_scores)
    assert summary["is_degenerate"]


def test_score_distribution_summary_normal_scores_not_degenerate():
    rng = np.random.default_rng(0)
    scores = rng.normal(size=200)
    summary = score_distribution_summary(scores)
    assert not summary["is_degenerate"]
    assert summary["min"] <= summary["p50"] <= summary["max"]


def test_top_n_novelty_table_returns_extremes():
    df = pd.DataFrame({"name": [f"n{i}" for i in range(20)]})
    scores = np.arange(20, dtype=float)
    table = top_n_novelty_table(df, scores, ["name"], n=3)
    top_scores = table[table["group"] == "top"]["novelty_score"].tolist()
    bottom_scores = table[table["group"] == "bottom"]["novelty_score"].tolist()
    assert sorted(top_scores, reverse=True) == [19.0, 18.0, 17.0]
    assert sorted(bottom_scores) == [0.0, 1.0, 2.0]


def test_novelty_by_segment_groups_correctly():
    df = pd.DataFrame({"segment": ["A", "A", "B", "B"]})
    scores = np.array([1.0, 3.0, 10.0, 10.0])
    out = novelty_by_segment(df, scores, "segment")
    assert out.loc["A", "mean"] == 2.0
    assert out.loc["B", "mean"] == 10.0
    assert out.loc["B", "std"] == 0.0


def test_customer_history_consistency_only_counts_repeat_customers():
    df = pd.DataFrame({"customer_id": ["c1", "c1", "c1", "c2"]})
    scores = np.array([1.0, 1.1, 0.9, 5.0])
    out = customer_history_consistency(df, scores, min_alerts=3)
    assert out["n_repeat_customers_evaluated"] == 1  # only c1 has >= 3 alerts


def test_customer_history_consistency_handles_no_repeat_customers():
    df = pd.DataFrame({"customer_id": ["c1", "c2", "c3"]})
    scores = np.array([1.0, 2.0, 3.0])
    out = customer_history_consistency(df, scores, min_alerts=2)
    assert out["n_repeat_customers_evaluated"] == 0
    assert out["median_within_customer_std"] is None


def test_ranking_stability_perfect_correlation():
    scores_a = np.array([1.0, 2.0, 3.0, 4.0])
    scores_b = np.array([10.0, 20.0, 30.0, 40.0])
    assert ranking_stability_between_models(scores_a, scores_b) == pytest.approx(1.0)


def test_ranking_stability_no_correlation():
    scores_a = np.array([1.0, 2.0, 3.0, 4.0])
    scores_b = np.array([4.0, 1.0, 3.0, 2.0])
    corr = ranking_stability_between_models(scores_a, scores_b)
    assert -1.0 <= corr <= 1.0


def test_exploratory_status_comparison_never_raises_and_is_grouped():
    df = pd.DataFrame({"status": ["Released", "Released", "UPS"]})
    scores = np.array([1.0, 2.0, 5.0])
    out = exploratory_status_comparison(df, scores, "status")
    assert set(out.index) == {"Released", "UPS"}
    assert out.loc["UPS", "mean"] == 5.0


def _fake_result(exp_id, degenerate=False, stability=0.5, within_std=1.0, overall_std=5.0, cost=1.0):
    from pipelines.entity.evaluation import ExperimentResult

    return ExperimentResult(
        experiment_id=exp_id,
        model_name="test_model",
        representation="test_repr",
        validation_scenario="test_scenario",
        distribution={"is_degenerate": degenerate, "std": overall_std},
        stability_spearman=stability,
        history_consistency={"median_within_customer_std": within_std, "overall_score_std": overall_std},
        fit_seconds=cost / 2,
        score_seconds=cost / 2,
    )


def test_select_champion_drops_degenerate_models():
    from pipelines.entity.evaluation import select_champion

    results = [
        _fake_result("degenerate", degenerate=True, stability=0.99),
        _fake_result("good", degenerate=False, stability=0.5),
    ]
    champion, rubric = select_champion(results)
    assert champion.experiment_id == "good"
    assert "degenerate" in rubric["dropped_degenerate_experiment_ids"]


def test_select_champion_prefers_higher_stability():
    from pipelines.entity.evaluation import select_champion

    results = [
        _fake_result("low_stability", stability=0.2),
        _fake_result("high_stability", stability=0.9),
    ]
    champion, _ = select_champion(results)
    assert champion.experiment_id == "high_stability"


def test_select_champion_tiebreaks_on_consistency_then_cost():
    from pipelines.entity.evaluation import select_champion

    results = [
        _fake_result("less_consistent", stability=0.5, within_std=4.0, overall_std=5.0, cost=1.0),
        _fake_result("more_consistent", stability=0.5, within_std=1.0, overall_std=5.0, cost=1.0),
    ]
    champion, _ = select_champion(results)
    assert champion.experiment_id == "more_consistent"


def test_select_champion_all_degenerate_raises():
    from pipelines.entity.evaluation import select_champion

    results = [_fake_result("d1", degenerate=True), _fake_result("d2", degenerate=True)]
    with pytest.raises(ValueError):
        select_champion(results)
