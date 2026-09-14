"""A Checkpoint: a trained weights file a Detector is loaded from."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    modified_at: float

    @property
    def version(self) -> str:
        """What Label Studio shows as the model version."""
        return self.path.name

    def supersedes(self, loaded: "Checkpoint | None") -> bool:
        """Whether a Detector loaded from `loaded` should be replaced by one loaded from this."""
        if loaded is None:
            return True
        return self.path != loaded.path or self.modified_at > loaded.modified_at
