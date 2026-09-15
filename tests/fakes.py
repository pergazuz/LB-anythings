"""In-memory implementations of the outbound ports. Hand-written, no mocking library."""

from collections.abc import Callable, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np

from lb_anythings.application.ports import (
    ConnectedModel,
    ExportedTask,
    Image,
    LabelStudioProject,
    LaunchFacts,
    ServiceCommand,
)
from lb_anythings.application.project_context import Credentials
from lb_anythings.domain.annotation import first_usable_annotation
from lb_anythings.domain.checkpoint import Checkpoint
from lb_anythings.domain.detection import Detection
from lb_anythings.domain.errors import (
    MediaUnavailable,
    ProjectExportFailed,
    ProjectWiringFailed,
    TrainingAlreadyActive,
)
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
        self.sizes: list[int] = []  # the Training Set size each run launched on
        self._status: dict[str, RunStatus] = {}

    def trained_at_size(self) -> int | None:
        return self.sizes[-1] if self.sizes else None

    def start(
        self, tracked_as: str | None = None, training_set_size: int | None = None
    ) -> TrainingRun:
        if self.active() is not None:
            raise TrainingAlreadyActive("a Training Run is already active")
        run = TrainingRun(f"run{len(self.runs) + 1}", float(len(self.runs) + 1), RunStatus.RUNNING)
        self.runs.append(run)
        self.tracked_as.append(tracked_as)
        if training_set_size is not None:
            self.sizes.append(training_set_size)
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
        self.prepared = 0
        self._run_id = run_id

    def prepare(self) -> None:
        self.prepared += 1

    def record_launch(self, facts: LaunchFacts) -> str | None:
        self.launches.append(facts)
        return self._run_id


class FakeClock:
    """Time that only moves when something sleeps, so a wait costs a test nothing."""

    def __init__(self) -> None:
        self.time = 0.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.time

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.time += seconds


class FakeHealthProbe:
    """Answers for the URLs it has been told about, after however many polls they need."""

    def __init__(self) -> None:
        self.polls: dict[str, int] = {}  # url -> polls still to go before it answers
        self.probed: list[str] = []

    def answering(self, url: str, after: int = 0) -> None:
        self.polls[url] = after

    def answers(self, url: str) -> bool:
        self.probed.append(url)
        remaining = self.polls.get(url)
        if remaining is None:
            return False
        if remaining <= 0:
            return True
        self.polls[url] = remaining - 1
        return False


class FakeService:
    def __init__(self, name: str, journal: list[str], *, alive: bool = True) -> None:
        self.name = name
        self._journal = journal
        self._alive = alive

    def running(self) -> bool:
        return self._alive

    def stop(self) -> None:
        if self._alive:
            self._journal.append(f"stop {self.name}")
        self._alive = False


class FakeServiceLauncher:
    """Launches nothing; tells the probe that what it launched now answers."""

    def __init__(self, probe: FakeHealthProbe, journal: list[str] | None = None) -> None:
        self.probe = probe
        self.journal = journal if journal is not None else []
        self.launched: list[ServiceCommand] = []
        self.answers_after: dict[str, int] = {}  # name -> polls before it answers
        self.never_answers: set[str] = set()
        self.dies: set[str] = set()  # names that exit without ever answering

    def launch(self, service: ServiceCommand) -> FakeService:
        self.launched.append(service)
        self.journal.append(f"launch {service.name}")
        if service.name in self.dies:
            return FakeService(service.name, self.journal, alive=False)
        if service.name not in self.never_answers:
            self.probe.answering(service.health_url, self.answers_after.get(service.name, 0))
        return FakeService(service.name, self.journal)


class FakeProjectAdmin:
    """Label Studio's project side, in a dict. Raises what the real one raises."""

    def __init__(self, journal: list[str] | None = None) -> None:
        self.journal = journal if journal is not None else []
        self.projects: dict[int, LabelStudioProject] = {}
        self.configs: dict[int, str] = {}
        self.models: dict[int, list[ConnectedModel]] = {}
        self.training_on_submit: list[int] = []
        self.fail_with: str | None = None

    def add(
        self, project_id: int, title: str, models: Sequence[ConnectedModel] = ()
    ) -> LabelStudioProject:
        project = LabelStudioProject(project_id, title)
        self.projects[project_id] = project
        self.models[project_id] = list(models)
        return project

    def project(self, project_id: int) -> LabelStudioProject | None:
        self._check()
        return self.projects.get(project_id)

    def project_titled(self, title: str) -> LabelStudioProject | None:
        self._check()
        return next((p for p in self.projects.values() if p.title == title), None)

    def create_project(self, title: str, label_config: str) -> LabelStudioProject:
        self._check()
        project = self.add(max(self.projects, default=0) + 1, title)
        self.configs[project.id] = label_config
        self.journal.append(f"create {title}")
        return project

    def connected_models(self, project_id: int) -> Sequence[ConnectedModel]:
        self._check()
        return list(self.models.get(project_id, []))

    def connect_model(self, project_id: int, url: str, title: str) -> None:
        self._check()
        self.models.setdefault(project_id, []).append(ConnectedModel(url, True))
        self.journal.append(f"connect {url}")

    def enable_training_on_submit(self, project_id: int) -> None:
        self._check()
        self.training_on_submit.append(project_id)
        self.journal.append(f"train-on-submit {project_id}")

    def _check(self) -> None:
        if self.fail_with:
            raise ProjectWiringFailed(self.fail_with)
