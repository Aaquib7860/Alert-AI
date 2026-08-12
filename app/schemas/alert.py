"""Phase 8 -- API request/response schemas.

Mirrors the example response shape in master plan section 13 exactly
(alert_id, alert_type, model_version, feature_version, schema_version,
novelty.global/customer, recommendation, reason_codes, historical_context,
scored_at), plus two documentation-only fields (`novelty_scale_note`,
`recommendation_threshold_note`) that state explicitly what the numbers
do and don't mean -- master plan: "Never describe an anomaly score as a
probability unless a separate calibration procedure proves that
interpretation."
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, Field

AlertType = Literal["customer_name", "transaction_name", "transaction_rule"]


class AlertScoreRequest(BaseModel):
    alert_id: str
    alert_type: AlertType
    raw_fields: dict[str, Any] = Field(
        description="Pre-decision fields only, matching the source sheet's "
        "non-leakage columns. See GET /api/v1/models/active for the exact "
        "field list per alert_type."
    )


class BatchScoreRequest(BaseModel):
    alerts: list[AlertScoreRequest]


class NoveltyScores(BaseModel):
    global_: float | None = Field(alias="global")
    customer: float | None

    model_config = {"populate_by_name": True}


class HistoricalContext(BaseModel):
    customer_prior_alert_count: int
    customer_baseline_known: bool


class AlertScoreResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    alert_id: str
    alert_type: AlertType
    model_version: str
    feature_version: str
    schema_version: str
    novelty: NoveltyScores
    novelty_scale_note: str
    recommendation: Literal["REVIEW", "LOWER_TOUCH_CANDIDATE"]
    recommendation_threshold_note: str
    plain_language_label: Literal["Needs Review", "Not Confident", "Looks Routine"] = Field(
        description="Everyday-language summary for UI display. Never says 'match'/'not a "
        "match' -- this describes the recommendation, not a sanctions-match determination. "
        "`recommendation` remains the authoritative field."
    )
    plain_language_detail: str
    reason_codes: list[str]
    reason_codes_plain: list[str] = Field(description="Plain-English translation of reason_codes, UI display only.")
    historical_context: HistoricalContext
    scored_at: datetime


class AlertScoreError(BaseModel):
    alert_id: str
    alert_type: AlertType
    error: str
    missing_required_columns: list[str] = []
    unexpected_columns: list[str] = []


class BatchScoreResponse(BaseModel):
    results: list[AlertScoreResponse]
    errors: list[AlertScoreError]
    n_requested: int
    n_scored: int
    n_failed: int


class ModelInfoResponse(BaseModel):
    model_config = {"protected_namespaces": ()}

    alert_type: AlertType
    model_version: str
    feature_version: str
    schema_version: str
    model_kind: str
    representation: str
    dataset_version: str
    n_reference_scores: int
    n_customer_baselines: int


class ActiveModelsResponse(BaseModel):
    models: list[ModelInfoResponse]
    updated_at: str


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    models_loaded: dict[str, bool]
    checked_at: datetime


class FeedbackRequest(BaseModel):
    model_config = {"protected_namespaces": ()}

    alert_id: str
    alert_type: AlertType
    model_version: str
    # Master plan section 5.1 outcome taxonomy -- deliberately not a bare
    # TRUE/FALSE (Phase 1 finding: UPS/Released are not trustworthy
    # ground truth without independent confirmation).
    compliance_outcome: Literal[
        "UNKNOWN", "HISTORICALLY_RELEASED", "CONFIRMED_FALSE_POSITIVE",
        "CONFIRMED_TRUE_MATCH", "ESCALATED", "FOLLOW_UP",
    ]
    notes: str | None = None


class FeedbackAck(BaseModel):
    received: bool
    alert_id: str
    recorded_at: datetime
