"""The Label Studio ML backend wire protocol, served by FastAPI."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from lb_anythings.adapters.inbound.http.schemas import PredictRequest, SetupRequest
from lb_anythings.application.project_context import Credentials
from lb_anythings.application.use_cases.predict_tasks import PredictionBatch, PredictTasks
from lb_anythings.application.use_cases.report_status import ReportStatus
from lb_anythings.application.use_cases.setup_project import SetupProject
from lb_anythings.domain.annotation_target import AnnotationTarget
from lb_anythings.domain.errors import InvalidLabelConfig
from lb_anythings.domain.prediction import Prediction
from lb_anythings.domain.task import Task


def create_app(
    *,
    report_status: ReportStatus,
    setup_project: SetupProject,
    predict_tasks: PredictTasks,
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
        tasks = [Task(id=t.get("id"), data=t.get("data") or {}) for t in body.tasks]
        return _serialize(predict_tasks(tasks, body.label_config, body.force_reload))

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
