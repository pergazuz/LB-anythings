"""Holds the one loaded Detector and swaps it when a newer Checkpoint appears."""

import logging
import threading
from collections.abc import Sequence

from lb_anythings.application.ports import (
    CheckpointRepository,
    Detector,
    DetectorFactory,
    Image,
)
from lb_anythings.domain.checkpoint import Checkpoint
from lb_anythings.domain.detection import Detection

logger = logging.getLogger(__name__)

NO_CHECKPOINT_VERSION = "none"


class NullDetector:
    """Serves while no Checkpoint exists: nothing is detected and the version says so."""

    version = NO_CHECKPOINT_VERSION

    def detect(self, image: Image) -> Sequence[Detection]:
        del image
        return ()


class DetectorCache:
    def __init__(self, factory: DetectorFactory, checkpoints: CheckpointRepository) -> None:
        self._factory = factory
        self._checkpoints = checkpoints
        self._lock = threading.Lock()
        self._loaded: Checkpoint | None = None
        self._detector: Detector = NullDetector()
        self._ever_loaded = False

    @property
    def serving_version(self) -> str:
        """The version of the Detector currently serving. Never loads anything."""
        with self._lock:
            return self._detector.version

    def current(self, *, force_reload: bool = False) -> Detector:
        """The Detector for the latest Checkpoint, reloading under a lock when it changed.

        Callers already inside detect() keep the Detector they were handed. A Checkpoint that
        cannot be loaded (a Training Run may be mid-write) leaves the serving Detector in
        place and is tried again next time.
        """
        latest = self._checkpoints.latest()
        with self._lock:
            if force_reload or not self._ever_loaded or self._changed(latest):
                try:
                    detector = self._factory.load(latest) if latest else NullDetector()
                except Exception:
                    logger.warning(
                        "could not load Checkpoint %s; keeping %s and retrying later",
                        latest.path if latest else None,
                        self._detector.version,
                        exc_info=True,
                    )
                    return self._detector
                self._detector = detector
                self._loaded = latest
                self._ever_loaded = True
                if latest is None:
                    logger.info("no Checkpoint found; serving empty Predictions until one exists")
                else:
                    logger.info("loaded Checkpoint %s (%s)", latest.version, latest.path)
            return self._detector

    def _changed(self, latest: Checkpoint | None) -> bool:
        if latest is None:
            return self._loaded is not None
        return latest.supersedes(self._loaded)
