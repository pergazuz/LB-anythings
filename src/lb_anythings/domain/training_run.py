"""Training Runs: one execution that consumes the Training Set and produces a Checkpoint."""

from dataclasses import dataclass
from enum import Enum, auto

from lb_anythings.domain.checkpoint import Checkpoint


class RunStatus(Enum):
    RUNNING = auto()
    SUCCEEDED = auto()
    FAILED = auto()


class RunTrigger(Enum):
    """What launched a Training Run. Known only to whatever launched it."""

    RETRAIN_THRESHOLD = "retrain-threshold"
    START_TRAINING = "start-training"
    COMMAND_LINE = "command-line"


@dataclass(frozen=True)
class TrainingRun:
    id: str
    started_at: float
    status: RunStatus
    checkpoint: Checkpoint | None = None

    @property
    def is_active(self) -> bool:
        return self.status is RunStatus.RUNNING


def derive_status(
    *, process_alive: bool, started_at: float, checkpoint: Checkpoint | None
) -> RunStatus:
    """Running while the process lives (the trainer rewrites its Checkpoint as it improves);
    once it is gone, succeeded if a Checkpoint newer than the start exists, else failed."""
    if process_alive:
        return RunStatus.RUNNING
    if checkpoint is not None and checkpoint.modified_at > started_at:
        return RunStatus.SUCCEEDED
    return RunStatus.FAILED
