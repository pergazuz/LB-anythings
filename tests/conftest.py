"""Fixtures for the primary seam: the app built by the composition root, driven over HTTP."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from fastapi.testclient import TestClient

from lb_anythings.bootstrap.container import Ports, build_app
from lb_anythings.bootstrap.settings import Settings
from lb_anythings.domain.checkpoint import Checkpoint
from lb_anythings.domain.detection import Detection
from tests.fakes import (
    FakeMediaResolver,
    InMemoryCheckpointRepository,
    InMemoryExampleStore,
    ScriptedDetectorFactory,
    SynchronousBackgroundRunner,
    image,
)

BEST = Path("best.pt")


@dataclass
class Fakes:
    """The outbound fakes behind one app, plus the one setup most tests share."""

    checkpoints: InMemoryCheckpointRepository
    detectors: ScriptedDetectorFactory
    media: FakeMediaResolver
    examples: InMemoryExampleStore

    def serve(self, detections: Sequence[Detection] = ()) -> None:
        """A Checkpoint whose Detector scripts `detections`, and an image at `a.jpg`."""
        self.checkpoints.checkpoint = Checkpoint(BEST, modified_at=1.0)
        self.detectors.scripts[BEST] = detections
        self.media.images["a.jpg"] = image(200, 100)

    @property
    def ports(self) -> Ports:
        return Ports(
            checkpoints=self.checkpoints,
            detector_factory=self.detectors,
            media=self.media,
            examples=self.examples,
            background=SynchronousBackgroundRunner(),
        )


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.chdir(tmp_path)  # an empty working directory: no `.env` can leak in
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture
def fakes() -> Fakes:
    return Fakes(
        InMemoryCheckpointRepository(),
        ScriptedDetectorFactory(),
        FakeMediaResolver(),
        InMemoryExampleStore(),
    )


@pytest.fixture
def client(settings: Settings, fakes: Fakes) -> Iterator[TestClient]:
    with TestClient(build_app(settings, ports=fakes.ports)) as c:
        yield c


@pytest.fixture
def png() -> bytes:
    """A 40x30 PNG. Needs the ML dependency group's decoder, so tests using it skip without."""
    cv2 = pytest.importorskip("cv2", reason="decoding needs the ML dependency group")
    ok, encoded = cv2.imencode(".png", np.zeros((30, 40, 3), dtype=np.uint8))
    assert ok
    return bytes(encoded)
