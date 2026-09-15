"""The retrain rule: when a saved Example should start a Training Run (pure domain seam).

The rule measures *growth since the last Training Run*, and the step it demands scales with the
Training Set, because accuracy improves as a power law in its size. See ADR 0004.
"""

import pytest

from lb_anythings.domain.retraining import RetrainPolicy, decide_retrain

POLICY = RetrainPolicy(threshold=25, minimum=4, growth=0.10)


def idle() -> bool:
    return False


def busy() -> bool:
    return True


def never_asked() -> bool:
    raise AssertionError("the decision was made without needing to know whether a run is active")


# --- the first Training Run ---


def test_a_training_set_that_has_never_been_trained_on_trains_now() -> None:
    """Examples brought across from elsewhere should not wait for more to arrive."""
    decision = decide_retrain(POLICY, training_set_size=333, trained_at_size=None, run_active=idle)

    assert decision.should_train and decision.reason == ""


def test_an_untrained_set_still_waits_for_the_minimum() -> None:
    decision = decide_retrain(
        POLICY, training_set_size=3, trained_at_size=None, run_active=never_asked
    )

    assert not decision.should_train
    assert "need at least 4" in decision.reason


def test_an_empty_training_set_says_so() -> None:
    decision = decide_retrain(
        POLICY, training_set_size=0, trained_at_size=None, run_active=never_asked
    )

    assert not decision.should_train and "empty" in decision.reason


# --- the floor, while the Training Set is small ---


def test_the_threshold_is_a_floor_while_the_proportion_would_be_smaller() -> None:
    """At 100 Examples, 10% is 10; the floor of 25 wins, so the next run is at 125."""
    assert RetrainPolicy(threshold=25, minimum=4, growth=0.10).step_after(100) == 25


def test_growth_below_the_floor_does_not_train_and_names_the_next_size() -> None:
    decision = decide_retrain(
        POLICY, training_set_size=110, trained_at_size=100, run_active=never_asked
    )

    assert not decision.should_train
    assert "next Training Run at 125" in decision.reason


def test_growth_that_meets_the_floor_trains() -> None:
    assert decide_retrain(
        POLICY, training_set_size=125, trained_at_size=100, run_active=idle
    ).should_train


def test_overshooting_the_step_still_trains() -> None:
    """Annotations arrive in bursts; the exact number is never landed on reliably."""
    assert decide_retrain(
        POLICY, training_set_size=131, trained_at_size=100, run_active=idle
    ).should_train


# --- the proportion, once the Training Set is large ---


@pytest.mark.parametrize(
    ("trained_at", "step"),
    [(50, 25), (250, 25), (251, 26), (500, 50), (1000, 100), (5000, 500)],
)
def test_the_step_grows_with_the_training_set(trained_at: int, step: int) -> None:
    assert POLICY.step_after(trained_at) == step


def test_a_large_set_is_not_retrained_for_a_handful_of_examples() -> None:
    """25 more Examples on 1000 is 2.5% and will not move mAP, but costs a whole run."""
    decision = decide_retrain(
        POLICY, training_set_size=1025, trained_at_size=1000, run_active=never_asked
    )

    assert not decision.should_train
    assert "next Training Run at 1100" in decision.reason


def test_the_reason_says_what_the_last_run_trained_on() -> None:
    decision = decide_retrain(
        POLICY, training_set_size=1025, trained_at_size=1000, run_active=never_asked
    )

    assert "100 new Examples after the 1000 the last run trained on" in decision.reason


# --- an active run ---


def test_no_second_run_starts_on_top_of_one_already_going() -> None:
    decision = decide_retrain(POLICY, training_set_size=125, trained_at_size=100, run_active=busy)

    assert not decision.should_train and "already active" in decision.reason


def test_a_trigger_declined_for_an_active_run_is_not_lost() -> None:
    """The old rule fired only on an exact multiple, so a busy moment cost a whole cycle."""
    declined = decide_retrain(POLICY, training_set_size=125, trained_at_size=100, run_active=busy)
    assert not declined.should_train

    next_annotation = decide_retrain(
        POLICY, training_set_size=126, trained_at_size=100, run_active=idle
    )

    assert next_annotation.should_train


# --- configuration ---


def test_turning_growth_off_leaves_a_fixed_step() -> None:
    fixed = RetrainPolicy(threshold=25, minimum=4, growth=0.0)

    assert fixed.step_after(10_000) == 25


def test_a_threshold_below_one_is_refused() -> None:
    with pytest.raises(ValueError, match="at least 1"):
        RetrainPolicy(threshold=0, minimum=4)


def test_negative_growth_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        RetrainPolicy(threshold=25, minimum=4, growth=-0.1)
