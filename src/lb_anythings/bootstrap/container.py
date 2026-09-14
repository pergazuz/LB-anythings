"""Composition root: wires settings and adapters into the application."""

from fastapi import FastAPI

from lb_anythings.adapters.inbound.http.app import create_app
from lb_anythings.application.project_context import ProjectContextHolder
from lb_anythings.application.use_cases.report_status import ReportStatus
from lb_anythings.application.use_cases.setup_project import SetupProject
from lb_anythings.bootstrap.settings import Settings


def build_app(settings: Settings) -> FastAPI:
    del settings  # the first setting the app itself reads arrives with the Detector
    context_holder = ProjectContextHolder()
    return create_app(
        report_status=ReportStatus(),
        setup_project=SetupProject(context_holder),
    )
