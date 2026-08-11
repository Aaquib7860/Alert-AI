"""Phase 6 -- Transaction / Rule anomaly models.

Master plan section 9 (Experimental Model Matrix), transaction rows T1-T5:
  T1: structured transaction features        -> Isolation Forest
  T2: behavioural representation              -> Isolation Forest
  T3: same representation                     -> Autoencoder
  T4: same representation                     -> LOF benchmark
  T5: same representation                     -> One-Class SVM benchmark

Model-fitting primitives (Isolation Forest, Autoencoder, LOF, OCSVM, SVD)
are reused from `pipelines.entity.anomaly_models` -- those functions are
pipeline-agnostic (fit on whatever matrix they're given, no entity-specific
assumption baked in) despite living under the `entity` package from when
Phase 5 wrote them first. This is deliberate code reuse of generic fitting
utilities, not a violation of the master plan's pipeline-separation rule --
that rule is about never sharing a *trained model instance* or *decision
logic* between Entity and Transaction (each pipeline still fits and owns
its own Isolation Forest/Autoencoder/etc. here), not about duplicating
identical scikit-learn/PyTorch boilerplate.

T1 ("structured transaction features") is the categorical + nationality
block only -- context features with no behavioural-history signal, the
Transaction-pipeline analogue of Entity's tabular-only E1 baseline.
T2-T5 ("behavioural representation", explicitly the *same* representation
across all four per master plan) use the full feature matrix -- context +
Beneficiary Name representation + the leakage-safe prior-alert/prior-rule-
diversity history features from Phase 4 -- SVD-reduced for T3-T5's dense
requirements, and reused as-is for T2 so all four experiments genuinely
share one representation.
"""
from __future__ import annotations

import scipy.sparse as sp

from features.transaction_features import TransactionFeatureArtifacts


def extract_structured_matrix(matrix: sp.csr_matrix, artifacts: TransactionFeatureArtifacts) -> sp.csr_matrix:
    """T1 slice: categorical + nationality blocks only (prefix of the full
    matrix, per transform_transaction_features' block order) -- excludes
    Beneficiary Name representation and the behavioural-history numeric
    features, which is exactly what makes this the "no history" baseline.
    """
    categorical_width = sum(
        sum(len(cats) for cats in enc.categories_)
        for enc in artifacts.categorical_encoders.values()
    )
    nationality_width = (
        sum(len(cats) for cats in artifacts.nationality_encoder.categories_)
        if artifacts.nationality_encoder is not None else 0
    )
    structured_width = categorical_width + nationality_width

    if matrix.shape[1] < structured_width:
        raise ValueError(
            f"Matrix has fewer columns ({matrix.shape[1]}) than the expected "
            f"structured-feature width ({structured_width}) -- artifacts/matrix mismatch"
        )
    return matrix[:, :structured_width]
