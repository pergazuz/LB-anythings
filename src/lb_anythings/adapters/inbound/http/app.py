"""The Label Studio ML backend wire protocol, served by FastAPI."""

import logging
import uuid
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from lb_anythings.adapters.inbound.http.schemas import (
    PredictRequest,
    SetupRequest,
    WebhookRequest,
)
from lb_anythings.application.ports import BackgroundRunner
from lb_anythings.application.project_context import Credentials
from lb_anythings.application.use_cases.ingest_annotation import AnnotationEvent, IngestAnnotation
from lb_anythings.application.use_cases.predict_tasks import PredictionBatch, PredictTasks
from lb_anythings.application.use_cases.report_status import ReportStatus
from lb_anythings.application.use_cases.setup_project import SetupProject
from lb_anythings.domain.annotation import parse_annotation
from lb_anythings.domain.annotation_target import AnnotationTarget
from lb_anythings.domain.errors import InvalidLabelConfig
from lb_anythings.domain.prediction import Prediction
from lb_anythings.domain.task import Task

logger = logging.getLogger(__name__)

ANNOTATION_ACTIONS = frozenset({"ANNOTATION_CREATED", "ANNOTATION_UPDATED"})


def create_app(
    *,
    background: BackgroundRunner,
    report_status: ReportStatus,
    setup_project: SetupProject,
    predict_tasks: PredictTasks,
    ingest_annotation: IngestAnnotation,
    on_startup: Callable[[], object] = lambda: None,
) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        on_startup()
        yield

    app = FastAPI(title="LB-anythings", lifespan=lifespan)

    @app.exception_handler(InvalidLabelConfig)
    def invalid_label_config(_: Request, exc: InvalidLabelConfig) -> JSONResponse:
        return JSONResponse({"detail": str(exc)}, status_code=400)

    @app.get("/")
    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "UP", "model_version": report_status().version}

    @app.post("/setup")
    def setup(body: SetupRequest) -> dict[str, str]:
        credentials = Credentials(body.hostname or None, body.access_token or None)
        setup_project(body.effective_label_config, credentials)
        return {"model_version": report_status().version}

    @app.post("/predict")
    def predict(body: PredictRequest) -> dict[str, Any]:
        tasks = [_task(t) for t in body.tasks]
        return _serialize(predict_tasks(tasks, body.label_config, body.force_reload))

    @app.post("/webhook")
    def webhook(body: WebhookRequest) -> JSONResponse:
        """Acknowledge at once; the work runs in the background. Label Studio ignores the body."""
        if body.action not in ANNOTATION_ACTIONS:
            return _skipped(f"action {body.action!r} is not handled")
        event = AnnotationEvent(
            task=_task(body.task or {}),
            annotation=parse_annotation(body.annotation or {}),
            label_config=body.effective_label_config,
        )
        if reason := ingest_annotation.rejection(event):
            return _skipped(reason)
        background.run(lambda: ingest_annotation(event))
        return JSONResponse({"job_id": uuid.uuid4().hex}, status_code=201)

    @app.get("/is_training")
    def is_training() -> dict[str, bool]:
        return {"is_training": report_status().is_training}

    @app.get("/metrics")
    def metrics() -> dict[str, Any]:
        return {}

    @app.post("/versions")
    def versions() -> dict[str, list[str]]:
        return {"versions": list(report_status().versions)}

    return app


def _skipped(reason: str) -> JSONResponse:
    logger.info("webhook skipped: %s", reason)
    return JSONResponse({"status": "skipped", "reason": reason}, status_code=201)


def _task(payload: dict[str, Any]) -> Task:
    return Task(id=payload.get("id"), data=payload.get("data") or {})


def _serialize(batch: PredictionBatch) -> dict[str, Any]:
    if batch.target is None:
        return {"results": [], "model_version": None}
    return {
        "results": [_serialize_prediction(p, batch.target) for p in batch.predictions],
        "model_version": batch.version,
    }


def _serialize_prediction(prediction: Prediction, target: AnnotationTarget) -> dict[str, Any]:
    return {
        "result": [
            {
                "from_name": target.control_name,
                "to_name": target.object_name,
                "type": "rectanglelabels",
                "value": {**region.box.to_label_studio(), "rectanglelabels": [region.label]},
                "score": region.score,
            }
            for region in prediction.regions
        ],
        "score": prediction.score,
        "model_version": prediction.version,
    }
