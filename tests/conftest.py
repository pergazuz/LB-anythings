"""Fixtures for the primary seam: the app built by the composition root, driven over HTTP."""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from lb_anythings.bootstrap.container import Ports, build_app
from lb_anythings.bootstrap.settings import Settings
from lb_anythings.domain.checkpoint import Checkpoint
from lb_anythings.domain.detection import Detection
from tests.fakes import (
    FakeMediaResolver,
    InMemoryCheckpointRepository,
    ScriptedDetectorFactory,
    image,
)

BEST = Path("best.pt")


@dataclass
class Fakes:
    """The outbound fakes behind one app, plus the one setup most predict tests share."""

    checkpoints: InMemoryCheckpointRepository
    detectors: ScriptedDetectorFactory
    media: FakeMediaResolver

    def serve(self, detections: Sequence[Detection] = ()) -> None:
        """A Checkpoint whose Detector scripts `detections`, and an image at `a.jpg`."""
        self.checkpoints.checkpoint = Checkpoint(BEST, modified_at=1.0)
        self.detectors.scripts[BEST] = detections
        self.media.images["a.jpg"] = image(200, 100)

    @property
    def ports(self) -> Ports:
        return Ports(
            checkpoints=self.checkpoints, detector_factory=self.detectors, media=self.media
        )


@pytest.fixture
def settings(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Settings:
    monkeypatch.chdir(tmp_path)  # an empty working directory: no `.env` can leak in
    return Settings(data_dir=tmp_path / "data")


@pytest.fixture
def fakes() -> Fakes:
    return Fakes(InMemoryCheckpointRepository(), ScriptedDetectorFactory(), FakeMediaResolver())


@pytest.fixture
def client(settings: Settings, fakes: Fakes) -> Iterator[TestClient]:
    with TestClient(build_app(settings, ports=fakes.ports)) as c:
        yield c
