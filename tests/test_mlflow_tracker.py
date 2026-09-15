"""Contract of the MLflow tracker, against a real SQLite store when MLflow is installed."""

from dataclasses import replace
from pathlib import Path

import pytest

from lb_anythings.adapters.outbound.mlflow.tracker import (
    MlflowExperimentTracker,
    NullExperimentTracker,
    TrackingConfig,
    launch_parameters,
    tracking_environment,
)
from lb_anythings.application.ports import LaunchFacts
from lb_anythings.domain.training_run import RunTrigger

THRESHOLD_LAUNCH = LaunchFacts(
    trigger=RunTrigger.RETRAIN_THRESHOLD,
    training_set_size=350,
    serving_version="best.pt@20260915T034500Z",
)
START_TRAINING_LAUNCH = LaunchFacts(
    trigger=RunTrigger.START_TRAINING,
    training_set_size=412,
    serving_version="none",
    exported_tasks=420,
    unannotated_tasks=6,
    uncollected_tasks=2,
)


def config(tmp_path: Path) -> TrackingConfig:
    return TrackingConfig(
        uri=f"sqlite:///{(tmp_path / 'mlflow.db').as_posix()}",
        artifact_dir=tmp_path / "artifacts",
        experiment="lb-anythings",
        run_name="active-20260915T034500Z",
    )


# --- what the Training Run is told ---


def test_the_callback_is_told_where_to_record_and_under_what_name(tmp_path: Path) -> None:
    environment = tracking_environment(config(tmp_path))

    assert environment["MLFLOW_TRACKING_URI"].startswith("sqlite:///")
    assert environment["MLFLOW_EXPERIMENT_NAME"] == "lb-anythings"
    assert environment["MLFLOW_RUN"] == "active-20260915T034500Z"
    assert "MLFLOW_RUN_ID" not in environment


def test_a_run_that_continues_a_recorded_one_says_which(tmp_path: Path) -> None:
    """The whole point: the launch facts and the metrics have to land on one row."""
    environment = tracking_environment(config(tmp_path), continues="abc123")

    assert environment["MLFLOW_RUN_ID"] == "abc123"


# --- what is recorded at launch ---


def test_the_launch_records_what_only_the_launcher_knows() -> None:
    parameters = launch_parameters(THRESHOLD_LAUNCH)

    assert parameters["trigger"] == "retrain-threshold"
    assert parameters["training_set_size"] == "350"
    assert parameters["serving_version"] == "best.pt@20260915T034500Z"


def test_the_export_counts_are_recorded_only_for_start_training() -> None:
    assert "exported_tasks" not in launch_parameters(THRESHOLD_LAUNCH)

    parameters = launch_parameters(START_TRAINING_LAUNCH)

    assert parameters["trigger"] == "start-training"
    assert parameters["exported_tasks"] == "420"
    assert parameters["unannotated_tasks"] == "6"
    assert parameters["uncollected_tasks"] == "2"


def test_recording_nothing_is_a_tracker_too() -> None:
    assert NullExperimentTracker().record_launch(THRESHOLD_LAUNCH) is None


def test_tracking_that_is_switched_off_records_nothing_and_says_so() -> None:
    assert MlflowExperimentTracker(lambda: None).record_launch(THRESHOLD_LAUNCH) is None


def test_a_broken_tracker_never_stops_a_training_run(tmp_path: Path) -> None:
    pytest.importorskip("mlflow", reason="tracking is an optional dependency group")
    unusable = TrackingConfig(
        uri="no-such-store://nowhere",  # fails fast; an unreachable host would retry for minutes
        artifact_dir=tmp_path / "artifacts",
        experiment="lb-anythings",
        run_name="active",
    )

    assert MlflowExperimentTracker(lambda: unusable).record_launch(THRESHOLD_LAUNCH) is None


# --- against a real store ---


def test_a_launch_is_recorded_and_can_be_read_back(tmp_path: Path) -> None:
    mlflow = pytest.importorskip("mlflow", reason="tracking is an optional dependency group")
    tracking = config(tmp_path)

    run_id = MlflowExperimentTracker(lambda: tracking).record_launch(START_TRAINING_LAUNCH)

    assert run_id is not None
    mlflow.set_tracking_uri(tracking.uri)
    recorded = mlflow.get_run(run_id)
    assert recorded.data.params["trigger"] == "start-training"
    assert recorded.data.params["training_set_size"] == "412"
    assert recorded.info.status == "FINISHED"  # so a dead Training Run leaves nothing running


def test_the_archive_is_where_the_data_directory_says(tmp_path: Path) -> None:
    """Left alone MLflow archives into ./mlruns, wherever the run happened to start."""
    mlflow = pytest.importorskip("mlflow", reason="tracking is an optional dependency group")
    tracking = config(tmp_path)

    MlflowExperimentTracker(lambda: tracking).record_launch(THRESHOLD_LAUNCH)

    mlflow.set_tracking_uri(tracking.uri)
    experiment = mlflow.get_experiment_by_name("lb-anythings")
    assert experiment.artifact_location == (tmp_path / "artifacts").as_uri()


def test_the_store_is_opened_before_anything_waits_on_it(tmp_path: Path) -> None:
    """The first recording costs seconds; paying it here keeps it off the retrain lock."""
    pytest.importorskip("mlflow", reason="tracking is an optional dependency group")
    tracking = config(tmp_path)
    tracker = MlflowExperimentTracker(lambda: tracking)

    tracker.prepare()

    assert (tmp_path / "mlflow.db").exists()
    assert (tmp_path / "artifacts").is_dir()


def test_preparing_a_store_that_cannot_be_opened_does_not_stop_the_service(
    tmp_path: Path,
) -> None:
    unusable = TrackingConfig("no-such-store://nowhere", tmp_path / "a", "lb-anythings", "active")

    MlflowExperimentTracker(lambda: unusable).prepare()  # must not raise

    NullExperimentTracker().prepare()


def test_the_run_is_told_to_sample_the_machine_it_trains_on(tmp_path: Path) -> None:
    """MLflow leaves system metrics off; training is the one thing here that loads a GPU."""
    environment = tracking_environment(config(tmp_path))

    assert environment["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] == "true"


def test_sampling_the_machine_can_be_turned_off(tmp_path: Path) -> None:
    quiet = replace(config(tmp_path), system_metrics=False)

    assert tracking_environment(quiet)["MLFLOW_ENABLE_SYSTEM_METRICS_LOGGING"] == "false"
