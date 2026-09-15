"""The retrain rule: when the Training Set's growth should start a Training Run on its own.

Growth, not size. Detector accuracy improves as a power law in the number of Examples, so a
fixed step buys less and less: 25 more Examples doubles a set of 25 and is a rounding error on
a set of 1000, while costing the same Training Run either way. The step therefore scales with
what is already there, with a floor for the early Examples that matter most. See ADR 0004.
"""

import math
from collections.abc import Callable
from dataclasses import dataclass


@dataclass(frozen=True)
class RetrainPolicy:
    threshold: int  # the Retrain Threshold: the fewest new Examples worth a Training Run
    minimum: int  # never train on fewer Examples than this in total
    growth: float = 0.10  # and never on fewer than this fraction of the Training Set

    def __post_init__(self) -> None:
        if self.threshold < 1:
            raise ValueError(f"the Retrain Threshold must be at least 1, got {self.threshold}")
        if self.growth < 0:
            raise ValueError(f"the growth fraction cannot be negative, got {self.growth}")

    def step_after(self, trained_at_size: int) -> int:
        """How many new Examples are worth a Training Run, given what the last one trained on."""
        return max(self.threshold, math.ceil(trained_at_size * self.growth))


@dataclass(frozen=True)
class RetrainDecision:
    should_train: bool
    reason: str = ""  # why not, when `should_train` is False


def decide_retrain(
    policy: RetrainPolicy,
    *,
    training_set_size: int,
    trained_at_size: int | None,
    run_active: Callable[[], bool],
) -> RetrainDecision:
    """Train once the Training Set has grown enough since the last Training Run was launched.

    `trained_at_size` is the size the last run launched on, or None when none ever has: a fresh
    install, or Examples brought across from elsewhere, should train on what is already there
    rather than wait for more.

    `run_active` is asked only when the growth warrants it: answering it may cost a pid check.
    Because the baseline is the last *launch*, declining for an active run does not lose the
    trigger -- the next Annotation asks again.
    """
    if training_set_size <= 0:
        return RetrainDecision(False, "the Training Set is empty")
    if training_set_size < policy.minimum:
        return RetrainDecision(
            False, f"Training Set has {training_set_size}; need at least {policy.minimum}"
        )
    if trained_at_size is not None:
        step = policy.step_after(trained_at_size)
        if training_set_size - trained_at_size < step:
            next_run_at = trained_at_size + step
            return RetrainDecision(
                False,
                f"Training Set has {training_set_size}; next Training Run at {next_run_at}"
                f" ({step} new Examples after the {trained_at_size} the last run trained on)",
            )
    if run_active():
        return RetrainDecision(False, "a Training Run is already active")
    return RetrainDecision(True)
