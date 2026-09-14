"""Composition root: wires settings and adapters into the application."""

from dataclasses import dataclass

from fastapi import FastAPI

from lb_anythings.adapters.inbound.http.app import create_app
from lb_anythings.adapters.outbound.background.runner import ThreadBackgroundRunner
from lb_anythings.adapters.outbound.filesystem.checkpoints import FilesystemCheckpointRepository
from lb_anythings.adapters.outbound.filesystem.examples import FilesystemExampleStore
from lb_anythings.adapters.outbound.labelstudio.media import LabelStudioMediaResolver
from lb_anythings.adapters.outbound.yolo.detector import YoloDetectorFactory
from lb_anythings.application.detector_cache import DetectorCache
from lb_anythings.application.ports import (
    BackgroundRunner,
    CheckpointRepository,
    DetectorFactory,
    ExampleStore,
    TaskMediaResolver,
)
from lb_anythings.application.project_context import Credentials, ProjectContextHolder
from lb_anythings.application.use_cases.ingest_annotation import IngestAnnotation
from lb_anythings.application.use_cases.predict_tasks import PredictTasks
from lb_anythings.application.use_cases.report_status import ReportStatus
from lb_anythings.application.use_cases.setup_project import SetupProject
from lb_anythings.bootstrap.settings import Settings


@dataclass(frozen=True)
class Ports:
    """The outbound adapters. Tests pass fakes; production_ports builds the real ones."""

    checkpoints: CheckpointRepository
    detector_factory: DetectorFactory
    media: TaskMediaResolver
    examples: ExampleStore
    background: BackgroundRunner


def production_ports(settings: Settings) -> Ports:
    return Ports(
        checkpoints=FilesystemCheckpointRepository(
            settings.trained_checkpoint, settings.checkpoint
        ),
        detector_factory=YoloDetectorFactory(conf=settings.conf, imgsz=settings.imgsz),
        media=LabelStudioMediaResolver(
            cache_dir=settings.data_dir / "cache",
            defaults=Credentials(
                hostname=settings.label_studio_url,
                access_token=(
                    settings.label_studio_api_key.get_secret_value()
                    if settings.label_studio_api_key
                    else None
                ),
            ),
        ),
        examples=FilesystemExampleStore(settings.data_dir / "examples"),
        background=ThreadBackgroundRunner(),
    )


def build_app(settings: Settings, ports: Ports | None = None) -> FastAPI:
    ports = ports or production_ports(settings)
    context_holder = ProjectContextHolder()
    detector_cache = DetectorCache(ports.detector_factory, ports.checkpoints)
    setup_project = SetupProject(context_holder)
    return create_app(
        background=ports.background,
        report_status=ReportStatus(detector_cache),
        setup_project=setup_project,
        predict_tasks=PredictTasks(setup_project, detector_cache, ports.media),
        ingest_annotation=IngestAnnotation(setup_project, ports.media, ports.examples),
        # Load (or note the absence of) the Checkpoint at startup, not on the first request.
        on_startup=lambda: detector_cache.current(),
    )
