"""Outbound ports: everything the application needs from the world, as Protocols."""

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
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

    def record_launch(self, facts: LaunchFacts) -> str | None:
        """Record what the launcher knows; return the id the Training Run continues under."""
        ...


class Trainer(Protocol):
    """Runs Training Runs. At most one is active at a time."""

    def start(self, tracked_as: str | None = None) -> TrainingRun:
        """Launch a Training Run, or raise TrainingAlreadyActive.

        `tracked_as` is the recorded run it continues, so the launch facts and the metrics the
        run itself produces end up on one row rather than two.
        """
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
