"""Phase 5 -- Entity anomaly / novelty models.

Master plan section 9 (Experimental Model Matrix), entity rows E1-E5:
  E1: baseline tabular/entity features         -> Isolation Forest
  E2: character/token entity representation    -> Isolation Forest
  E3: same representation                       -> Autoencoder
  E4: same representation                       -> LOF benchmark
  E5: same representation                       -> One-Class SVM benchmark

"Isolation Forest is the starting candidate because the current dataset is
small-to-medium, tabular, heterogeneous, unlabeled, and cost-sensitive.
This is not a pre-decided final winner." (master plan section 4)

No model here ever reads a label. Every `fit_*` function trains only on
`X_train`; scoring functions run independently on whatever `X` is passed.
Champion selection (see `pipelines/entity/evaluation.py`) never compares
against Status/UPS/Released -- master plan section 11: "the agent must not
report generic classification accuracy."
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from sklearn.decomposition import TruncatedSVD
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM

from features.entity_features import EntityFeatureArtifacts

RANDOM_STATE = 42


# ---------------------------------------------------------------------------
# E1: tabular-only baseline (excludes the character/token name representation)
# ---------------------------------------------------------------------------

def tabular_block_slice(artifacts: EntityFeatureArtifacts) -> slice:
    """Column slice for the tabular-only feature block, computed from the
    fitted artifacts' own known output widths -- not by guessing indices.
    Depends on `transform_entity_features`'s block order (name vectorizers,
    then categorical encoders, then nationality encoders, then numeric);
    asserts that order explicitly so a future reordering fails loudly here
    rather than silently slicing the wrong columns.
    """
    name_width = sum(len(vec.vocabulary_) for vec in artifacts.name_vectorizers.values())
    tabular_start = name_width
    if artifacts.feature_names:
        assert all(b.startswith("name_tfidf::") for b in artifacts.feature_names[: len(artifacts.name_vectorizers)]), (
            "Expected name_tfidf blocks first -- tabular_block_slice depends on this order"
        )
    return slice(tabular_start, None)  # everything after the name blocks


def extract_tabular_matrix(matrix: sp.csr_matrix, artifacts: EntityFeatureArtifacts) -> sp.csr_matrix:
    return matrix[:, tabular_block_slice(artifacts)]


def extract_name_representation_matrix(matrix: sp.csr_matrix, artifacts: EntityFeatureArtifacts) -> sp.csr_matrix:
    name_width = sum(len(vec.vocabulary_) for vec in artifacts.name_vectorizers.values())
    return matrix[:, :name_width]


# ---------------------------------------------------------------------------
# Dimensionality reduction for the character/token representation (E2-E5)
# ---------------------------------------------------------------------------

def fit_svd(X_train: sp.csr_matrix, n_components: int = 50) -> TruncatedSVD:
    """TruncatedSVD on the sparse TF-IDF name representation -- a standard,
    explainable, non-LLM dimensionality reduction (not a decision engine),
    needed to make Autoencoder/LOF/OCSVM tractable on a representation that
    would otherwise be thousands of mostly-zero dimensions.
    """
    n_components = min(n_components, X_train.shape[1] - 1, X_train.shape[0] - 1)
    svd = TruncatedSVD(n_components=n_components, random_state=RANDOM_STATE)
    svd.fit(X_train)
    return svd


# ---------------------------------------------------------------------------
# Isolation Forest (E1, E2)
# ---------------------------------------------------------------------------

def fit_isolation_forest(X_train) -> IsolationForest:
    model = IsolationForest(
        n_estimators=200, contamination="auto", random_state=RANDOM_STATE, n_jobs=-1,
    )
    model.fit(X_train)
    return model


def score_isolation_forest(model: IsolationForest, X) -> np.ndarray:
    # score_samples: higher = more normal. Flip sign so higher = more
    # anomalous, consistent with every other score function in this module.
    return -model.score_samples(X)


# ---------------------------------------------------------------------------
# Autoencoder (E3)
# ---------------------------------------------------------------------------

class _Autoencoder(nn.Module):
    def __init__(self, input_dim: int, bottleneck: int = 8):
        super().__init__()
        hidden = max(bottleneck * 2, min(32, input_dim // 2))
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, bottleneck), nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.Linear(bottleneck, hidden), nn.ReLU(),
            nn.Linear(hidden, input_dim),
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


@dataclass
class AutoencoderModel:
    net: _Autoencoder
    input_mean: np.ndarray
    input_std: np.ndarray
    train_losses: list[float]


def fit_autoencoder(
    X_train: np.ndarray, bottleneck: int = 8, epochs: int = 100, lr: float = 1e-3,
) -> AutoencoderModel:
    """Trains on X_train only. Standardizes using train-only mean/std (a
    holdout scored later reuses these exact statistics -- computing new
    stats from the holdout would leak its distribution into scoring).
    """
    torch.manual_seed(RANDOM_STATE)
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)
    std[std == 0] = 1.0  # avoid divide-by-zero for constant columns
    X_norm = (X_train - mean) / std

    net = _Autoencoder(input_dim=X_train.shape[1], bottleneck=bottleneck)
    optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    X_tensor = torch.tensor(X_norm, dtype=torch.float32)
    losses = []
    net.train()
    for _ in range(epochs):
        optimizer.zero_grad()
        reconstructed = net(X_tensor)
        loss = loss_fn(reconstructed, X_tensor)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    return AutoencoderModel(net=net, input_mean=mean, input_std=std, train_losses=losses)


def score_autoencoder(model: AutoencoderModel, X: np.ndarray) -> np.ndarray:
    X_norm = (X - model.input_mean) / model.input_std
    X_tensor = torch.tensor(X_norm, dtype=torch.float32)
    model.net.eval()
    with torch.no_grad():
        reconstructed = model.net(X_tensor)
        error = ((X_tensor - reconstructed) ** 2).mean(dim=1).numpy()
    return error  # higher reconstruction error = more anomalous


# ---------------------------------------------------------------------------
# LOF benchmark (E4)
# ---------------------------------------------------------------------------

def fit_lof(X_train: np.ndarray, n_neighbors: int = 20) -> LocalOutlierFactor:
    n_neighbors = min(n_neighbors, max(1, X_train.shape[0] - 1))
    model = LocalOutlierFactor(n_neighbors=n_neighbors, novelty=True)
    model.fit(X_train)
    return model


def score_lof(model: LocalOutlierFactor, X: np.ndarray) -> np.ndarray:
    return -model.score_samples(X)


# ---------------------------------------------------------------------------
# One-Class SVM benchmark (E5)
# ---------------------------------------------------------------------------

def fit_ocsvm(X_train: np.ndarray, nu: float = 0.1) -> OneClassSVM:
    model = OneClassSVM(kernel="rbf", nu=nu, gamma="scale")
    model.fit(X_train)
    return model


def score_ocsvm(model: OneClassSVM, X: np.ndarray) -> np.ndarray:
    return -model.score_samples(X)
