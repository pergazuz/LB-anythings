"""Checkpoints on disk: the newest of the trained one and the configured one serves."""

from pathlib import Path

from lb_anythings.domain.checkpoint import Checkpoint


class FilesystemCheckpointRepository:
    def __init__(self, trained: Path, configured: Path | None) -> None:
        # Order matters only for ties: the trained Checkpoint wins them.
        self._candidates = [p for p in (trained, configured) if p is not None]

    def latest(self) -> Checkpoint | None:
        existing = [Checkpoint(p, p.stat().st_mtime) for p in self._candidates if p.is_file()]
        if not existing:
            return None
        return max(existing, key=lambda checkpoint: checkpoint.modified_at)
