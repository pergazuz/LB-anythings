"""Contract of the tracking config the composition root builds."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from lb_anythings.bootstrap.container import tracking_for
from lb_anythings.bootstrap.settings import Settings

AT = datetime(2026, 9, 15, 3, 45, 0, tzinfo=UTC)


def test_tracking_is_off_when_the_operator_turns_it_off() -> None:
    assert tracking_for(Settings(data_dir=Path("data"), tracking=False), AT) is None


def test_a_recorded_run_is_named_for_the_moment_it_started() -> None:
    tracking = tracking_for(Settings(data_dir=Path("data"), train_run_name="active"), AT)

    assert tracking is not None
    assert tracking.run_name == "active-20260915T034500Z"


def test_runs_are_recorded_in_sqlite_under_the_data_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """MLflow's file store is deprecated and refuses to open; SQLite is the supported local one."""
    monkeypatch.chdir(tmp_path)

    tracking = tracking_for(Settings(data_dir=Path("data")), AT)

    assert tracking is not None
    assert tracking.uri == f"sqlite:///{(tmp_path / 'data/mlflow/mlflow.db').resolve().as_posix()}"
    assert tracking.experiment == "lb-anythings"


def test_the_archive_is_absolute_so_it_does_not_follow_the_working_directory(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)

    tracking = tracking_for(Settings(data_dir=Path("data")), AT)

    assert tracking is not None
    assert tracking.artifact_dir == (tmp_path / "data/mlflow/artifacts").resolve()


def test_the_operator_can_record_runs_somewhere_else() -> None:
    settings = Settings(data_dir=Path("data"), tracking_uri="http://mlflow.local:5000")

    tracking = tracking_for(settings, AT)

    assert tracking is not None
    assert tracking.uri == "http://mlflow.local:5000"
