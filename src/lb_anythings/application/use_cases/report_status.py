"""What the Operator and Label Studio can ask about the backend's state."""

from dataclasses import dataclass

from lb_anythings.application.detector_cache import NO_CHECKPOINT_VERSION, DetectorCache
from lb_anythings.application.ports import Trainer


@dataclass(frozen=True)
class Status:
    version: str
    is_training: bool = False
    versions: tuple[str, ...] = ()


class ReportStatus:
    def __init__(self, detector_cache: DetectorCache, trainer: Trainer) -> None:
        self._detector_cache = detector_cache
        self._trainer = trainer

    def __call__(self) -> Status:
        version = self._detector_cache.serving_version
        known = () if version == NO_CHECKPOINT_VERSION else (version,)
        return Status(
            version=version, is_training=self._trainer.active() is not None, versions=known
        )
