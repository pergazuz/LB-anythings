"""Outbound ports: everything the application needs from the world, as Protocols."""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from lb_anythings.application.project_context import Credentials
from lb_anythings.domain.checkpoint import Checkpoint
from lb_anythings.domain.detection import Detection


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
