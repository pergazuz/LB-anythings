"""A Checkpoint: a trained weights file a Detector is loaded from."""

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path


@dataclass(frozen=True)
class Checkpoint:
    path: Path
    modified_at: float

    @property
    def version(self) -> str:
        """What Label Studio shows as the model version.

        Every Training Run writes the same file name, so the modification time is part of
        the version: each run's output is a distinct version, and a reload is visible.
        """
        stamp = datetime.fromtimestamp(self.modified_at, tz=UTC).strftime("%Y%m%dT%H%M%SZ")
        return f"{self.path.name}@{stamp}"

    def supersedes(self, loaded: "Checkpoint | None") -> bool:
        """Whether a Detector loaded from `loaded` should be replaced by one loaded from this."""
        if loaded is None:
            return True
        return self.path != loaded.path or self.modified_at > loaded.modified_at
