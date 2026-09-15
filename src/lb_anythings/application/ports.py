"""Outbound ports: everything the application needs from the world, as Protocols."""

from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from lb_anythings.application.project_context import Credentials
from lb_anythings.domain.annotation import Annotation
from lb_anythings.domain.checkpoint import Checkpoint
from lb_anythings.domain.detection import Detection
from lb_anythings.domain.example import Example
from lb_anythings.domain.task import Task
from lb_anythings.domain.training_run import RunTrigger, TrainingRun


@dataclass(frozen=True)
class Image:
    """A decoded image as it crosses ports: height x width x 3, BGR."""

    pixels: NDArray[np.uint8]

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])


class Detector(Protocol):
    @property
    def version(self) -> str: ...

    def detect(self, image: Image) -> Sequence[Detection]: ...


class DetectorFactory(Protocol):
    def load(self, checkpoint: Checkpoint) -> Detector: ...


class CheckpointRepository(Protocol):
    def latest(self) -> Checkpoint | None: ...


class TaskMediaResolver(Protocol):
    def load(self, reference: str, credentials: Credentials) -> Image:
        """Return the decoded image, or raise MediaUnavailable."""
        ...


class ExampleStore(Protocol):
    """Where the Training Set accumulates. Saving an Example for a Task again replaces it."""

    def save(self, example: Example, image: Image) -> None: ...

    def positive_count(self) -> int: ...

    def all(self) -> Sequence[Example]: ...


class BackgroundRunner(Protocol):
    """Runs a job after the HTTP response has gone out. A thread in production."""

    def run(self, job: Callable[[], object]) -> None: ...


@dataclass(frozen=True)
class LaunchFacts:
    """What is true of a Training Run when it is launched, and knowable only to its launcher.

    The Training Set size is the one that matters: quality plotted against how much has been
    labelled is what says whether labelling is still worth doing. The Detector cannot know it,
    so nothing downstream of the launch can record it.
    """

    trigger: RunTrigger
    training_set_size: int
    serving_version: str  # the Checkpoint this run is trying to beat
    exported_tasks: int | None = None  # Start Training only: what the project export held
    unannotated_tasks: int | None = None
    uncollected_tasks: int | None = None


class ExperimentTracker(Protocol):
    """Records Training Runs. The no-op default keeps the service running untracked."""

    def prepare(self) -> None:
        """Do whatever the first recording would otherwise pay for, before anything waits.

        Recording a launch happens between deciding to train and launching, under the lock
        that keeps two Annotations from starting two runs, so it must be quick by then.
        """
        ...

    def record_launch(self, facts: LaunchFacts) -> str | None:
        """Record what the launcher knows; return the id the Training Run continues under."""
        ...


class Trainer(Protocol):
    """Runs Training Runs. At most one is active at a time."""

    def start(
        self, tracked_as: str | None = None, training_set_size: int | None = None
    ) -> TrainingRun:
        """Launch a Training Run, or raise TrainingAlreadyActive.

        `tracked_as` is the recorded run it continues, so the launch facts and the metrics the
        run itself produces end up on one row rather than two. `training_set_size` is how many
        Examples it is training on, kept as the baseline the next retrain decision measures
        growth against.
        """
        ...

    def trained_at_size(self) -> int | None:
        """The Training Set size the last Training Run launched on, or None if none has."""
        ...

    def active(self) -> TrainingRun | None:
        """The running Training Run with its current status, or None."""
        ...

    def refresh(self, run: TrainingRun) -> TrainingRun: ...


@dataclass(frozen=True)
class ExportedTask:
    """A Task as a project export lists it, with its first non-cancelled Annotation if any."""

    task: Task
    annotation: Annotation | None


class LabelStudioProjectClient(Protocol):
    def exported_tasks(self, project_id: int, credentials: Credentials) -> Sequence[ExportedTask]:
        """Every Task of the project, each with its Annotation if it has a usable one.

        Raises ProjectExportFailed when Label Studio does not hand the tasks over.
        """
        ...


class FrameSource(Protocol):
    """A video opened for mining: its length, its size, and its frames."""

    @property
    def frame_count(self) -> int: ...

    @property
    def size(self) -> tuple[int, int]:
        """Width and height in pixels."""
        ...

    def frames(self, stride: int) -> Iterator[tuple[int, Image]]:
        """Every `stride`-th frame with its index, from the start."""
        ...

    def frame_at(self, index: int) -> Image | None:
        """One frame by index, or None when it cannot be read."""
        ...


@dataclass(frozen=True)
class ServiceCommand:
    """One long-running Service of the Stack: what to run, and where it answers when it is up."""

    name: str
    command: tuple[str, ...]
    url: str  # what an Operator opens
    health_path: str = "/health"
    environment: Mapping[str, str] = field(default_factory=dict)
    # Whether stopping it should stop what it spawned. True for a Service with
    # workers; false for one whose children are meant to outlive it.
    stop_descendants: bool = True

    @property
    def health_url(self) -> str:
        return f"{self.url.rstrip('/')}{self.health_path}"


class ServiceProcess(Protocol):
    """A Service this process started, and is therefore responsible for stopping."""

    def running(self) -> bool: ...

    def stop(self) -> None:
        """Stop it and everything it spawned. Stopping one already stopped does nothing."""
        ...


class ServiceLauncher(Protocol):
    def launch(self, service: ServiceCommand) -> ServiceProcess: ...


class HealthProbe(Protocol):
    def answers(self, url: str) -> bool:
        """Whether anything answered at all. A Service still starting up does not."""
        ...


@dataclass(frozen=True)
class LabelStudioProject:
    id: int
    title: str


@dataclass(frozen=True)
class ConnectedModel:
    """A model Label Studio already has on a project."""

    url: str
    interactive: bool  # whether it is asked for a Prediction while a Task is open


class LabelStudioProjectAdmin(Protocol):
    """The project side of Label Studio's API: what connecting a project by hand would do.

    Every method raises ProjectWiringFailed when Label Studio will not play along.
    """

    def project(self, project_id: int) -> LabelStudioProject | None: ...

    def project_titled(self, title: str) -> LabelStudioProject | None:
        """The project with exactly this title, or None. The first, if somehow there are two."""
        ...

    def create_project(self, title: str, label_config: str) -> LabelStudioProject: ...

    def connected_models(self, project_id: int) -> Sequence[ConnectedModel]: ...

    def connect_model(self, project_id: int, url: str, title: str) -> None:
        """Connect this backend as the project's model, with interactive Predictions on.

        Label Studio health-checks the URL and calls its `/setup` before accepting it, so the
        backend has to be answering already.
        """
        ...

    def enable_training_on_submit(self, project_id: int) -> None:
        """Switch on the toggle without which no Annotation ever reaches the backend."""
        ...
