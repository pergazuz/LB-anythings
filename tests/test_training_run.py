"""Training Run status: the pure rule behind `/is_training`."""

from pathlib import Path

from lb_anythings.domain.checkpoint import Checkpoint
from lb_anythings.domain.training_run import RunStatus, derive_status

BEST = Path("best.pt")
NEWER = Checkpoint(BEST, modified_at=200.0)
OLDER = Checkpoint(BEST, modified_at=50.0)


def test_a_run_is_running_while_its_process_lives_whatever_the_checkpoint_says() -> None:
    # ultralytics rewrites best.pt during training, so a newer file proves nothing yet
    assert (
        derive_status(process_alive=True, started_at=100.0, checkpoint=NEWER) is RunStatus.RUNNING
    )
    assert derive_status(process_alive=True, started_at=100.0, checkpoint=None) is RunStatus.RUNNING


def test_a_finished_run_succeeded_when_a_checkpoint_newer_than_its_start_exists() -> None:
    status = derive_status(process_alive=False, started_at=100.0, checkpoint=NEWER)

    assert status is RunStatus.SUCCEEDED


def test_a_finished_run_failed_when_no_newer_checkpoint_exists() -> None:
    assert (
        derive_status(process_alive=False, started_at=100.0, checkpoint=OLDER) is RunStatus.FAILED
    )
    assert derive_status(process_alive=False, started_at=100.0, checkpoint=None) is RunStatus.FAILED


def test_a_checkpoints_version_names_the_file_and_when_it_was_written() -> None:
    assert Checkpoint(Path("runs/active/weights/best.pt"), 1_000.0).version == (
        "best.pt@19700101T001640Z"
    )
