"""Composition root: wires settings and adapters into the application."""

import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fastapi import FastAPI

from lb_anythings.adapters.inbound.http.app import create_app
from lb_anythings.adapters.outbound.background.runner import ThreadBackgroundRunner
from lb_anythings.adapters.outbound.filesystem.checkpoints import FilesystemCheckpointRepository
from lb_anythings.adapters.outbound.filesystem.examples import FilesystemExampleStore
from lb_anythings.adapters.outbound.labelstudio.media import LabelStudioMediaResolver
from lb_anythings.adapters.outbound.labelstudio.project import LabelStudioExportClient
from lb_anythings.adapters.outbound.mlflow.tracker import (
    MlflowExperimentTracker,
    NullExperimentTracker,
    TrackingConfig,
    tracking_environment,
)
from lb_anythings.adapters.outbound.opencv.frames import OpenCvFrameSource, write_frames
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
    ExperimentTracker,
    LabelStudioProjectClient,
    TaskMediaResolver,
    Trainer,
)
from lb_anythings.application.project_context import Credentials, ProjectContextHolder
from lb_anythings.application.use_cases.ingest_annotation import IngestAnnotation
from lb_anythings.application.use_cases.mine_hard_frames import (
    MineHardFrames,
    MiningParameters,
    MiningProgress,
)
from lb_anythings.application.use_cases.predict_tasks import PredictTasks
from lb_anythings.application.use_cases.report_status import ReportStatus
from lb_anythings.application.use_cases.setup_project import SetupProject
from lb_anythings.application.use_cases.train_on_project import TrainOnProject
from lb_anythings.bootstrap.settings import Settings
from lb_anythings.domain.errors import NoCheckpointAvailable
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
    tracker: ExperimentTracker


def configured_credentials(settings: Settings) -> Credentials:
    """Label Studio's URL and token from settings; setup's, when sent, take precedence."""
    api_key = settings.label_studio_api_key
    return Credentials(
        hostname=settings.label_studio_url,
        access_token=api_key.get_secret_value() if api_key else None,
    )


def checkpoint_repository(settings: Settings) -> FilesystemCheckpointRepository:
    """Where every Detector looks for its Checkpoint: the trained one, else the configured."""
    return FilesystemCheckpointRepository(settings.trained_checkpoint, settings.checkpoint)


def spawn_environment(settings: Settings) -> dict[str, str]:
    """The paths a spawned Training Run must not work out for itself.

    `Settings` has already resolved these against the directory the server started in. Passing
    them means the child writes there whatever directory it inherits and whatever a `.env`
    beside it says.
    """
    environment = {"LB_DATA_DIR": str(settings.data_dir)}
    if settings.checkpoint is not None:
        environment["LB_CHECKPOINT"] = str(settings.checkpoint)
    return environment


def production_ports(settings: Settings) -> Ports:
    return Ports(
        checkpoints=checkpoint_repository(settings),
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
            environment=spawn_environment(settings),
        ),
        project_client=LabelStudioExportClient(),
        tracker=MlflowExperimentTracker(lambda: tracking_for(settings))
        if settings.tracking
        else NullExperimentTracker(),
    )


def ensure_tracking_environment(tracking: TrackingConfig, tracked_as: str | None) -> bool:
    """Put what ultralytics' MLflow callback reads into this process' environment.

    The Training Run is its own process and exits after training, so its environment is ours
    to set.
    """
    os.environ.update(tracking_environment(tracking, tracked_as))
    return True


def tracking_for(settings: Settings, started_at: datetime | None = None) -> TrackingConfig | None:
    """Where this Training Run is recorded, or None when tracking is off.

    The run name carries a UTC stamp because the run *folder* is always the same one: every
    recorded run would otherwise be called `active` and the history would be unreadable.
    """
    if not settings.tracking:
        return None
    root = settings.tracking_dir  # absolute already: Settings resolves once, at startup
    stamp = (started_at or datetime.now(UTC)).strftime("%Y%m%dT%H%M%SZ")
    return TrackingConfig(
        uri=settings.tracking_uri or f"sqlite:///{(root / 'mlflow.db').as_posix()}",
        artifact_dir=root / "artifacts",
        experiment=settings.tracking_experiment,
        run_name=f"{settings.train_run_name}-{stamp}",
    )


def train_with_yolo(settings: Settings, tracked_as: str | None = None) -> TrainingReport:
    """One Training Run in this process: what `lb-anythings train` does.

    `tracked_as` names the recorded run the launcher already opened, so the launch facts and
    the metrics this run produces land on one row. Run by hand there is no launcher, and
    ultralytics' callback opens a run of its own.
    """
    tracking = tracking_for(settings)
    tracked = tracking is not None and ensure_tracking_environment(tracking, tracked_as)
    store = FilesystemExampleStore(settings.examples_dir)
    return run_training(
        tracked=tracked,
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


def mine_with_yolo(
    settings: Settings,
    video: Path,
    parameters: MiningParameters,
    on_progress: Callable[[MiningProgress], None] = lambda _: None,
) -> list[Path]:
    """Mine a video for Hard Frames and write them: what `lb-anythings mine` does."""
    checkpoint = checkpoint_repository(settings).latest()
    if checkpoint is None:
        raise NoCheckpointAvailable(
            "nothing to mine with: train a Checkpoint first, or point LB_CHECKPOINT at one"
        )
    # A lower confidence floor than Predictions use: mining is interested in borderline boxes.
    detector = YoloDetectorFactory(conf=settings.mine_conf, imgsz=settings.imgsz).load(checkpoint)
    miner = MineHardFrames(detector, on_progress)
    with OpenCvFrameSource(video) as source:
        picks = miner(source, parameters)
    return write_frames(picks, settings.hard_frames_dir)


def ready_before_serving(
    detector_cache: DetectorCache, tracker: ExperimentTracker
) -> Callable[[], None]:
    """Pay the slow costs at startup: the Checkpoint load, and opening the tracking store.

    Left until first use, the one lands on the first Prediction and the other on the lock that
    serialises the retrain decision, where it would delay the Training Run it is recording.
    """

    def ready() -> None:
        detector_cache.current()
        tracker.prepare()

    return ready


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
            ports.tracker,
            lambda: detector_cache.serving_version,
        ),
        train_on_project=TrainOnProject(
            setup_project,
            ports.project_client,
            recorder,
            ports.examples,
            ports.trainer,
            ports.tracker,
            lambda: detector_cache.serving_version,
            minimum_examples=settings.min_examples,
            configured_credentials=configured_credentials(settings),
        ),
        on_startup=ready_before_serving(detector_cache, ports.tracker),
    )
