"""The Label Studio ML backend wire protocol, served by FastAPI."""

from typing import Any

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from lb_anythings.adapters.inbound.http.schemas import PredictRequest, SetupRequest
from lb_anythings.application.project_context import Credentials
from lb_anythings.application.use_cases.report_status import ReportStatus
from lb_anythings.application.use_cases.setup_project import SetupProject
from lb_anythings.domain.errors import InvalidLabelConfig


def create_app(*, report_status: ReportStatus, setup_project: SetupProject) -> FastAPI:
    app = FastAPI(title="LB-anythings")

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
        # Predictions arrive with the Detector (next ticket); until then there is nothing to say.
        del body
        return {"results": [], "model_version": None}

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
