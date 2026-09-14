"""In-memory implementations of the outbound ports. Hand-written, no mocking library."""

from collections.abc import Callable, Sequence
from pathlib import Path

import numpy as np

from lb_anythings.application.ports import Image
from lb_anythings.application.project_context import Credentials
from lb_anythings.domain.checkpoint import Checkpoint
from lb_anythings.domain.detection import Detection
from lb_anythings.domain.errors import MediaUnavailable
from lb_anythings.domain.example import Example


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

    def load(self, checkpoint: Checkpoint) -> ScriptedDetector:
        self.loads.append(checkpoint.path)
        count = self.loads.count(checkpoint.path)
        return ScriptedDetector(
            f"{checkpoint.version}#{count}", self.scripts.get(checkpoint.path, ())
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
