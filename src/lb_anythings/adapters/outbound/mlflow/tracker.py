"""Records Training Runs with MLflow. Every piece of MLflow knowledge lives in this module.

A Training Run is recorded twice over, from two processes: the launcher writes what only it
knows (what triggered the run, how large the Training Set was) and the run itself writes its
metrics and archives its Checkpoint through ultralytics' own MLflow callback. They land on one
row because the launcher hands the recorded run's id down, and MLflow resumes a finished run.
"""

import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from lb_anythings.application.ports import LaunchFacts

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrackingConfig:
    """Where Training Runs are recorded."""

    uri: str  # an MLflow tracking URI: a local SQLite file, or a server
    artifact_dir: Path  # where a run's Checkpoints are archived
    experiment: str
    run_name: str


def tracking_environment(tracking: TrackingConfig, continues: str | None = None) -> dict[str, str]:
    """The variables ultralytics' MLflow callback reads.

    With `continues` set the callback resumes that recorded run instead of opening a new one,
    which is what keeps a launch and its metrics on one row.
    """
    environment = {
        "MLFLOW_TRACKING_URI": tracking.uri,
        "MLFLOW_EXPERIMENT_NAME": tracking.experiment,
        "MLFLOW_RUN": tracking.run_name,
        "MLFLOW_DISABLE_AGENT_HINT": "1",  # keeps a notice for coding agents out of the run log
    }
    if continues is not None:
        environment["MLFLOW_RUN_ID"] = continues
    return environment


def ensure_experiment(tracking: TrackingConfig) -> bool:
    """Point MLflow at the store and create the experiment, saying where its artifacts go.

    Left to itself MLflow writes artifacts under `./mlruns`, relative to whatever directory the
    Training Run was spawned in -- the same trap as ultralytics' relative `project`. Creation is
    the one moment the location can be set. False when MLflow is not installed.
    """
    # Set before the import, which is when MLflow decides whether to print it: a notice aimed
    # at coding agents has no business in the log an Operator is told to read.
    os.environ.setdefault("MLFLOW_DISABLE_AGENT_HINT", "1")
    try:
        import mlflow  # deferred: the tracking dependency group is optional
    except ImportError:
        return False
    tracking.artifact_dir.mkdir(parents=True, exist_ok=True)
    mlflow.set_tracking_uri(tracking.uri)
    if mlflow.get_experiment_by_name(tracking.experiment) is None:
        mlflow.create_experiment(
            tracking.experiment, artifact_location=tracking.artifact_dir.as_uri()
        )
    return True


def launch_parameters(facts: LaunchFacts) -> dict[str, str]:
    """The launch facts as MLflow parameters, omitting the ones that do not apply."""
    parameters = {
        "trigger": facts.trigger.value,
        "training_set_size": str(facts.training_set_size),
        "serving_version": facts.serving_version,
    }
    optional = {
        "exported_tasks": facts.exported_tasks,
        "unannotated_tasks": facts.unannotated_tasks,
        "uncollected_tasks": facts.uncollected_tasks,
    }
    parameters.update({name: str(value) for name, value in optional.items() if value is not None})
    return parameters


class NullExperimentTracker:
    """Records nothing. What the service uses when tracking is off."""

    def prepare(self) -> None:
        return None

    def record_launch(self, facts: LaunchFacts) -> str | None:
        return None


class MlflowExperimentTracker:
    """Opens the recorded run at launch and closes it again, leaving it to be resumed.

    The run is finished rather than left open: a Training Run whose process dies would
    otherwise leave a run marked running for ever, and MLflow resumes a finished run happily.
    """

    def __init__(self, configure: Callable[[], TrackingConfig | None]) -> None:
        self._configure = configure

    def prepare(self) -> None:
        """Import MLflow and open the store now.

        The first recording costs seconds -- the import, and creating the store's schema --
        and it would otherwise be paid while holding the lock that serialises the retrain
        decision, delaying the Training Run it is recording. The Checkpoint is loaded eagerly
        at startup for the same reason.
        """
        tracking = self._configure()
        if tracking is None:
            return
        try:
            ensure_experiment(tracking)
        except Exception:  # noqa: BLE001 - tracking must never stop the service starting
            logger.warning("could not open the tracking store", exc_info=True)

    def record_launch(self, facts: LaunchFacts) -> str | None:
        tracking = self._configure()
        if tracking is None:
            return None
        try:
            if not ensure_experiment(tracking):  # inside the guard: opening a store can fail
                return None
            import mlflow  # ensure_experiment answering True means this import works

            mlflow.set_experiment(tracking.experiment)
            with mlflow.start_run(run_name=tracking.run_name) as run:
                mlflow.log_params(launch_parameters(facts))
                return str(run.info.run_id)
        except Exception:  # noqa: BLE001 - a tracking failure must never stop a Training Run
            logger.warning("could not record the Training Run's launch", exc_info=True)
            return None
