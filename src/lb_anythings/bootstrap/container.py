"""Composition root: wires settings and adapters into the application."""

import sys
from dataclasses import dataclass

from fastapi import FastAPI

from lb_anythings.adapters.inbound.http.app import create_app
from lb_anythings.adapters.outbound.background.runner import ThreadBackgroundRunner
from lb_anythings.adapters.outbound.filesystem.checkpoints import FilesystemCheckpointRepository
from lb_anythings.adapters.outbound.filesystem.examples import FilesystemExampleStore
from lb_anythings.adapters.outbound.labelstudio.media import LabelStudioMediaResolver
from lb_anythings.adapters.outbound.labelstudio.project import LabelStudioExportClient
from lb_anythings.adapters.outbound.subprocess.trainer import SubprocessTrainer
from lb_anythings.adapters.outbound.yolo.detector import YoloDetectorFactory
from lb_anythings.adapters.outbound.yolo.training import (
    TrainingConfig,
    TrainingReport,
    run_training,
)
from lb_anythings.application.detector_cache import DetectorCache
from lb_anythings.application.example_recording import ExampleRecorder
from lb_anythings.application.ports import (
    BackgroundRunner,
    CheckpointRepository,
    DetectorFactory,
    ExampleStore,
    LabelStudioProjectClient,
    TaskMediaResolver,
    Trainer,
)
from lb_anythings.application.project_context import Credentials, ProjectContextHolder
from lb_anythings.application.use_cases.ingest_annotation import IngestAnnotation
from lb_anythings.application.use_cases.predict_tasks import PredictTasks
from lb_anythings.application.use_cases.report_status import ReportStatus
from lb_anythings.application.use_cases.setup_project import SetupProject
from lb_anythings.application.use_cases.train_on_project import TrainOnProject
from lb_anythings.bootstrap.settings import Settings
from lb_anythings.domain.retraining import RetrainPolicy


@dataclass(frozen=True)
class Ports:
    """The outbound adapters. Tests pass fakes; production_ports builds the real ones."""

    checkpoints: CheckpointRepository
    detector_factory: DetectorFactory
    media: TaskMediaResolver
    examples: ExampleStore
    background: BackgroundRunner
    trainer: Trainer
    project_client: LabelStudioProjectClient


def configured_credentials(settings: Settings) -> Credentials:
    """Label Studio's URL and token from settings; setup's, when sent, take precedence."""
    api_key = settings.label_studio_api_key
    return Credentials(
        hostname=settings.label_studio_url,
        access_token=api_key.get_secret_value() if api_key else None,
    )


def production_ports(settings: Settings) -> Ports:
    return Ports(
        checkpoints=FilesystemCheckpointRepository(
            settings.trained_checkpoint, settings.checkpoint
        ),
        detector_factory=YoloDetectorFactory(conf=settings.conf, imgsz=settings.imgsz),
        media=LabelStudioMediaResolver(
            cache_dir=settings.cache_dir, defaults=configured_credentials(settings)
        ),
        examples=FilesystemExampleStore(settings.examples_dir),
        background=ThreadBackgroundRunner(),
        trainer=SubprocessTrainer(
            # The same interpreter and environment, so the child reads the same settings.
            command=[sys.executable, "-m", "lb_anythings", "train"],
            runs_dir=settings.runs_dir,
            run_name=settings.train_run_name,
            checkpoint=settings.trained_checkpoint,
        ),
        project_client=LabelStudioExportClient(),
    )


def train_with_yolo(settings: Settings) -> TrainingReport:
    """One Training Run in this process: what `lb-anythings train` does."""
    store = FilesystemExampleStore(settings.examples_dir)
    return run_training(
        examples=store.training_files(),
        class_names=store.class_names(),
        layout_root=settings.dataset_dir,
        runs_root=settings.runs_dir,
        config=TrainingConfig(
            base_model=settings.train_base_model,
            epochs=settings.train_epochs,
            patience=settings.train_patience,
            imgsz=settings.imgsz,
            batch=settings.train_batch,
            device=settings.device,
            lr0=settings.train_lr0,
            run_name=settings.train_run_name,
            min_examples=settings.min_examples,
        ),
    )


def build_app(settings: Settings, ports: Ports | None = None) -> FastAPI:
    ports = ports or production_ports(settings)
    context_holder = ProjectContextHolder()
    detector_cache = DetectorCache(ports.detector_factory, ports.checkpoints)
    setup_project = SetupProject(context_holder)
    recorder = ExampleRecorder(ports.media, ports.examples)
    return create_app(
        background=ports.background,
        report_status=ReportStatus(detector_cache, ports.trainer),
        setup_project=setup_project,
        predict_tasks=PredictTasks(setup_project, detector_cache, ports.media),
        ingest_annotation=IngestAnnotation(
            setup_project,
            recorder,
            ports.examples,
            ports.trainer,
            RetrainPolicy(threshold=settings.retrain_every, minimum=settings.min_examples),
        ),
        train_on_project=TrainOnProject(
            setup_project,
            ports.project_client,
            recorder,
            ports.examples,
            ports.trainer,
            minimum_examples=settings.min_examples,
            configured_credentials=configured_credentials(settings),
        ),
        # Load (or note the absence of) the Checkpoint at startup, not on the first request.
        on_startup=lambda: detector_cache.current(),
    )
