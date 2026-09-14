"""Contract of the filesystem Checkpoint repository."""

import os
from pathlib import Path

from lb_anythings.adapters.outbound.filesystem.checkpoints import FilesystemCheckpointRepository


def _file(path: Path, *, modified_at: float) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"weights")
    os.utime(path, (modified_at, modified_at))
    return path


def test_no_files_means_no_checkpoint(tmp_path: Path) -> None:
    repo = FilesystemCheckpointRepository(tmp_path / "runs/active/weights/best.pt", None)

    assert repo.latest() is None


def test_the_configured_checkpoint_serves_when_nothing_was_trained(tmp_path: Path) -> None:
    configured = _file(tmp_path / "custom.pt", modified_at=1_000)
    repo = FilesystemCheckpointRepository(tmp_path / "runs/active/weights/best.pt", configured)

    checkpoint = repo.latest()

    assert checkpoint is not None
    assert (checkpoint.path, checkpoint.version, checkpoint.modified_at) == (
        configured,
        "custom.pt",
        1_000,
    )


def test_the_newer_of_trained_and_configured_serves(tmp_path: Path) -> None:
    configured = _file(tmp_path / "custom.pt", modified_at=1_000)
    trained = _file(tmp_path / "runs/active/weights/best.pt", modified_at=2_000)
    repo = FilesystemCheckpointRepository(trained, configured)

    checkpoint = repo.latest()
    assert checkpoint is not None and checkpoint.path == trained

    os.utime(configured, (3_000, 3_000))  # the Operator drops in a fresher file
    checkpoint = repo.latest()
    assert checkpoint is not None and checkpoint.path == configured


def test_a_missing_configured_path_is_ignored(tmp_path: Path) -> None:
    repo = FilesystemCheckpointRepository(
        tmp_path / "runs/active/weights/best.pt", tmp_path / "gone.pt"
    )

    assert repo.latest() is None
