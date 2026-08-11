import numpy as np
import pytest
import scipy.sparse as sp

from pipelines.entity.anomaly_models import (
    fit_autoencoder,
    fit_isolation_forest,
    fit_lof,
    fit_ocsvm,
    fit_svd,
    score_autoencoder,
    score_isolation_forest,
    score_lof,
    score_ocsvm,
)


@pytest.fixture
def toy_dense_data():
    rng = np.random.default_rng(0)
    normal = rng.normal(loc=0, scale=1, size=(200, 10))
    return normal


def test_isolation_forest_fit_score_shapes(toy_dense_data):
    model = fit_isolation_forest(toy_dense_data)
    scores = score_isolation_forest(model, toy_dense_data)
    assert scores.shape[0] == toy_dense_data.shape[0]
    assert np.isfinite(scores).all()


def test_isolation_forest_accepts_sparse_input():
    X = sp.random(100, 20, density=0.3, format="csr", random_state=0)
    model = fit_isolation_forest(X)
    scores = score_isolation_forest(model, X)
    assert scores.shape[0] == 100


def test_isolation_forest_ranks_obvious_outlier_higher(toy_dense_data):
    X = toy_dense_data.copy()
    X_with_outlier = np.vstack([X, np.full((1, X.shape[1]), 100.0)])
    model = fit_isolation_forest(X_with_outlier)
    scores = score_isolation_forest(model, X_with_outlier)
    assert scores[-1] == scores.max()  # the injected extreme row scores most anomalous


def test_autoencoder_fit_score_shapes(toy_dense_data):
    model = fit_autoencoder(toy_dense_data, bottleneck=4, epochs=20)
    scores = score_autoencoder(model, toy_dense_data)
    assert scores.shape[0] == toy_dense_data.shape[0]
    assert np.isfinite(scores).all()
    assert (scores >= 0).all()  # MSE reconstruction error is non-negative


def test_autoencoder_loss_decreases_during_training(toy_dense_data):
    model = fit_autoencoder(toy_dense_data, bottleneck=4, epochs=50)
    # loss should generally trend down -- compare first vs last quartile mean
    first_quarter = np.mean(model.train_losses[:10])
    last_quarter = np.mean(model.train_losses[-10:])
    assert last_quarter < first_quarter


def test_autoencoder_ranks_obvious_outlier_higher(toy_dense_data):
    X = toy_dense_data.copy()
    X_with_outlier = np.vstack([X, np.full((1, X.shape[1]), 100.0)])
    model = fit_autoencoder(X_with_outlier, bottleneck=4, epochs=80)
    scores = score_autoencoder(model, X_with_outlier)
    assert scores[-1] == scores.max()


def test_lof_fit_score_shapes(toy_dense_data):
    model = fit_lof(toy_dense_data)
    scores = score_lof(model, toy_dense_data)
    assert scores.shape[0] == toy_dense_data.shape[0]
    assert np.isfinite(scores).all()


def test_lof_handles_small_train_set_gracefully():
    X = np.random.default_rng(0).normal(size=(5, 3))
    model = fit_lof(X, n_neighbors=20)  # more neighbors requested than rows available
    scores = score_lof(model, X)
    assert scores.shape[0] == 5


def test_ocsvm_fit_score_shapes(toy_dense_data):
    model = fit_ocsvm(toy_dense_data)
    scores = score_ocsvm(model, toy_dense_data)
    assert scores.shape[0] == toy_dense_data.shape[0]
    assert np.isfinite(scores).all()


def test_svd_reduces_dimensionality():
    X = sp.random(100, 500, density=0.05, format="csr", random_state=0)
    svd = fit_svd(X, n_components=20)
    reduced = svd.transform(X)
    assert reduced.shape == (100, 20)


def test_svd_caps_components_to_available_dimensions():
    X = sp.random(5, 3, density=0.5, format="csr", random_state=0)
    svd = fit_svd(X, n_components=50)  # requesting more components than possible
    reduced = svd.transform(X)
    assert reduced.shape[1] < 50
