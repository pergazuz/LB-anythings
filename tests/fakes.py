"""In-memory implementations of the outbound ports. Hand-written, no mocking library."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from lb_anythings.application.ports import ExportedTask, Image, LaunchFacts
from lb_anythings.application.project_context import Credentials
from lb_anythings.domain.annotation import first_usable_annotation
from lb_anythings.domain.checkpoint import Checkpoint
from lb_anythings.domain.detection import Detection
from lb_anythings.domain.errors import MediaUnavailable, ProjectExportFailed, TrainingAlreadyActive
from lb_anythings.domain.example import Example
from lb_anythings.domain.task import Task
from lb_anythings.domain.training_run import RunStatus, TrainingRun


def image(width: int, height: int) -> Image:
    return Image(np.zeros((height, width, 3), dtype=np.uint8))


class InMemoryCheckpointRepository:
    def __init__(self, checkpoint: Checkpoint | None = None) -> None:
        self.checkpoint = checkpoint

    def latest(self) -> Checkpoint | None:
        return self.checkpoint


class ScriptedDetector:
    def __init__(self, version: str, detections: Sequence[Detection]) -> None:
        self.version = version
        self._detections = tuple(detections)

    def detect(self, image: Image) -> Sequence[Detection]:
        del image
        return self._detections


class ScriptedDetectorFactory:
    """Loads a ScriptedDetector per Checkpoint path; the version records how often it loaded."""

    def __init__(self) -> None:
        self.scripts: dict[Path, Sequence[Detection]] = {}
        self.loads: list[Path] = []
        self.fail_next_load = False  # a Checkpoint mid-write: newer on disk, not yet loadable

    def load(self, checkpoint: Checkpoint) -> ScriptedDetector:
        if self.fail_next_load:
            self.fail_next_load = False
            raise RuntimeError("weights file is truncated")
        self.loads.append(checkpoint.path)
        count = self.loads.count(checkpoint.path)
        return ScriptedDetector(
            f"{checkpoint.path.name}#{count}", self.scripts.get(checkpoint.path, ())
        )


class FakeMediaResolver:
    def __init__(self) -> None:
        self.images: dict[str, Image] = {}
        self.requests: list[tuple[str, Credentials]] = []

    def load(self, reference: str, credentials: Credentials) -> Image:
        self.requests.append((reference, credentials))
        try:
            return self.images[reference]
        except KeyError:
            raise MediaUnavailable(f"no such image: {reference}") from None


class InMemoryExampleStore:
    def __init__(self) -> None:
        self._examples: dict[str, tuple[Example, Image]] = {}

    def save(self, example: Example, image: Image) -> None:
        self._examples[example.task_id] = (example, image)

    def positive_count(self) -> int:
        return sum(1 for example, _ in self._examples.values() if example.is_positive)

    def all(self) -> Sequence[Example]:
        return [example for example, _ in self._examples.values()]


class SynchronousBackgroundRunner:
    """Runs the job before returning, so a test sees its effects right after the request."""

    def run(self, job: Callable[[], object]) -> None:
        job()


class FakeTrainer:
    """Records Training Runs; a test flips a run's status with finish()."""

    def __init__(self) -> None:
        self.runs: list[TrainingRun] = []
        self.tracked_as: list[str | None] = []  # what each run was told to continue
        self._status: dict[str, RunStatus] = {}

    def start(self, tracked_as: str | None = None) -> TrainingRun:
        if self.active() is not None:
            raise TrainingAlreadyActive("a Training Run is already active")
        run = TrainingRun(f"run{len(self.runs) + 1}", float(len(self.runs) + 1), RunStatus.RUNNING)
        self.runs.append(run)
        self.tracked_as.append(tracked_as)
        self._status[run.id] = RunStatus.RUNNING
        return run

    def active(self) -> TrainingRun | None:
        for run in self.runs:
            if self._status[run.id] is RunStatus.RUNNING:
                return self.refresh(run)
        return None

    def refresh(self, run: TrainingRun) -> TrainingRun:
        return replace(run, status=self._status[run.id])

    def finish(self, run: TrainingRun, *, succeeded: bool) -> None:
        self._status[run.id] = RunStatus.SUCCEEDED if succeeded else RunStatus.FAILED


class ScriptedProjectClient:
    """Answers a project export from raw Label Studio task dicts scripted per project id."""

    def __init__(self) -> None:
        self.tasks: dict[int, list[dict[str, Any]]] = {}
        self.requests: list[tuple[int, Credentials]] = []
        self.fail_with: str | None = None

    def exported_tasks(self, project_id: int, credentials: Credentials) -> Sequence[ExportedTask]:
        self.requests.append((project_id, credentials))
        if self.fail_with:
            raise ProjectExportFailed(self.fail_with)
        return [
            ExportedTask(
                Task(id=item.get("id"), data=item.get("data") or {}),
                first_usable_annotation(item.get("annotations") or []),
            )
            for item in self.tasks.get(project_id, [])
        ]


class FakeExperimentTracker:
    """Remembers every launch it was told about, and hands back an id to continue."""

    def __init__(self, run_id: str | None = "recorded-run") -> None:
        self.launches: list[LaunchFacts] = []
        self._run_id = run_id

    def record_launch(self, facts: LaunchFacts) -> str | None:
        self.launches.append(facts)
        return self._run_id
