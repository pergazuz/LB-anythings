"""The retrain rule: when the Training Set's growth should start a Training Run on its own."""

from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrainPolicy:
    threshold: int  # the Retrain Threshold: a Training Run every this many positive Examples
    minimum: int  # never train on fewer than this

    def __post_init__(self) -> None:
        if self.threshold < 1:
            raise ValueError(f"the Retrain Threshold must be at least 1, got {self.threshold}")


@dataclass(frozen=True)
class RetrainDecision:
    should_train: bool
    reason: str = ""  # why not, when `should_train` is False


def decide_retrain(
    policy: RetrainPolicy, *, training_set_size: int, run_active: Callable[[], bool]
) -> RetrainDecision:
    """Train at every multiple of the threshold once the minimum is met and no run is active.

    `run_active` is asked only when the size warrants it: answering it may cost a pid check.
    """
    if training_set_size <= 0:
        return RetrainDecision(False, "the Training Set is empty")
    if training_set_size < policy.minimum:
        return RetrainDecision(
            False, f"Training Set has {training_set_size}; need at least {policy.minimum}"
        )
    if training_set_size % policy.threshold != 0:
        next_run_at = (training_set_size // policy.threshold + 1) * policy.threshold
        return RetrainDecision(
            False, f"Training Set has {training_set_size}; next Training Run at {next_run_at}"
        )
    if run_active():
        return RetrainDecision(False, "a Training Run is already active")
    return RetrainDecision(True)
