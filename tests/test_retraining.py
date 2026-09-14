"""The retrain rule: when a saved Example should start a Training Run (pure domain seam)."""

import pytest

from lb_anythings.domain.retraining import RetrainPolicy, decide_retrain

POLICY = RetrainPolicy(threshold=25, minimum=4)


def idle() -> bool:
    return False


def busy() -> bool:
    return True


def test_a_training_set_at_a_multiple_of_the_threshold_trains() -> None:
    decision = decide_retrain(POLICY, training_set_size=25, run_active=idle)

    assert decision.should_train and decision.reason == ""


def test_later_multiples_train_again() -> None:
    assert decide_retrain(POLICY, training_set_size=50, run_active=idle).should_train


def test_between_multiples_nothing_happens_and_the_next_threshold_is_named() -> None:
    decision = decide_retrain(POLICY, training_set_size=26, run_active=idle)

    assert not decision.should_train
    assert "next Training Run at 50" in decision.reason


def test_below_the_minimum_nothing_happens_even_at_a_multiple() -> None:
    policy = RetrainPolicy(threshold=2, minimum=4)

    decision = decide_retrain(policy, training_set_size=2, run_active=idle)

    assert not decision.should_train
    assert "need at least 4" in decision.reason


def test_an_active_run_blocks_a_new_one() -> None:
    decision = decide_retrain(POLICY, training_set_size=25, run_active=busy)

    assert not decision.should_train
    assert "already active" in decision.reason


def test_whether_a_run_is_active_is_only_asked_at_a_threshold() -> None:
    asked: list[bool] = []

    def run_active() -> bool:
        asked.append(True)
        return False

    decide_retrain(POLICY, training_set_size=26, run_active=run_active)

    assert asked == []


def test_zero_examples_never_train() -> None:
    policy = RetrainPolicy(threshold=1, minimum=1)

    assert not decide_retrain(policy, training_set_size=0, run_active=idle).should_train


def test_a_threshold_below_one_is_rejected() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        RetrainPolicy(threshold=0, minimum=1)
