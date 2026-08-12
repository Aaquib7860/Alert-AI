"""Phase 8 -- API routes. Master plan section 13 API contract."""
from __future__ import annotations

from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException

from app.schemas.alert import (
    ActiveModelsResponse,
    AlertScoreError,
    AlertScoreRequest,
    AlertScoreResponse,
    BatchScoreRequest,
    BatchScoreResponse,
    FeedbackAck,
    FeedbackRequest,
    HealthResponse,
    ModelInfoResponse,
)
from app.services.feedback import record_feedback
from app.services.model_info import find_model_info_by_version, get_all_active_model_info, models_health
from app.services.model_registry import ModelNotActiveError
from app.services.scoring import AlertValidationError, score_alert
from app.services.shadow_log import record_shadow_score

router = APIRouter(prefix="/api/v1")


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    loaded = models_health()
    status = "ok" if all(loaded.values()) else "degraded"
    return HealthResponse(status=status, models_loaded=loaded, checked_at=datetime.now(timezone.utc))


@router.get("/models/active", response_model=ActiveModelsResponse)
def active_models() -> ActiveModelsResponse:
    infos = get_all_active_model_info()
    if not infos:
        raise HTTPException(status_code=503, detail="No active models registered. Run scripts/build_scoring_registry.py")
    return ActiveModelsResponse(
        models=[ModelInfoResponse(**i) for i in infos],
        updated_at=datetime.now(timezone.utc).isoformat(),
    )


@router.get("/models/{model_version}", response_model=ModelInfoResponse)
def model_by_version(model_version: str) -> ModelInfoResponse:
    info = find_model_info_by_version(model_version)
    if info is None:
        raise HTTPException(status_code=404, detail=f"No active model with version {model_version!r}")
    return ModelInfoResponse(**info)


@router.post("/alerts/score", response_model=AlertScoreResponse)
def score_single_alert(request: AlertScoreRequest) -> AlertScoreResponse:
    try:
        result = score_alert(request.alert_type, request.alert_id, request.raw_fields)
    except AlertValidationError as e:
        raise HTTPException(status_code=422, detail={
            "error": str(e),
            "missing_required_columns": e.missing_required_columns,
            "unexpected_columns": e.unexpected_columns,
        })
    except ModelNotActiveError as e:
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    # Shadow mode (master plan section 10): every real score gets an audit
    # record, no downstream action is ever triggered by it -- this line
    # writes the log entry and nothing else.
    record_shadow_score(result, source="api")
    return AlertScoreResponse(**result)


@router.post("/alerts/batch-score", response_model=BatchScoreResponse)
def score_batch(request: BatchScoreRequest) -> BatchScoreResponse:
    results: list[AlertScoreResponse] = []
    errors: list[AlertScoreError] = []

    for alert in request.alerts:
        try:
            result = score_alert(alert.alert_type, alert.alert_id, alert.raw_fields)
            record_shadow_score(result, source="api_batch")
            results.append(AlertScoreResponse(**result))
        except AlertValidationError as e:
            errors.append(AlertScoreError(
                alert_id=alert.alert_id, alert_type=alert.alert_type, error=str(e),
                missing_required_columns=e.missing_required_columns,
                unexpected_columns=e.unexpected_columns,
            ))
        except (ModelNotActiveError, ValueError) as e:
            errors.append(AlertScoreError(alert_id=alert.alert_id, alert_type=alert.alert_type, error=str(e)))

    return BatchScoreResponse(
        results=results, errors=errors,
        n_requested=len(request.alerts), n_scored=len(results), n_failed=len(errors),
    )


@router.post("/feedback", response_model=FeedbackAck)
def submit_feedback(request: FeedbackRequest) -> FeedbackAck:
    entry = record_feedback(
        request.alert_id, request.alert_type, request.model_version,
        request.compliance_outcome, request.notes,
    )
    return FeedbackAck(received=True, alert_id=request.alert_id, recorded_at=entry["recorded_at"])
