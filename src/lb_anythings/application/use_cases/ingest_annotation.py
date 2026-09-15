"""Ingest an Annotation: what an Annotator's submit or update turns into."""

import logging
import threading
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

from lb_anythings.application.example_recording import ExampleRecorder
from lb_anythings.application.ports import (
    ExampleStore,
    ExperimentTracker,
    LaunchFacts,
    Trainer,
)
from lb_anythings.application.use_cases.setup_project import SetupProject
from lb_anythings.domain.annotation import Annotation
from lb_anythings.domain.errors import TrainingAlreadyActive
from lb_anythings.domain.retraining import RetrainDecision, RetrainPolicy, decide_retrain
from lb_anythings.domain.task import Task
from lb_anythings.domain.training_run import RunTrigger

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AnnotationEvent:
    task: Task
    annotation: Annotation
    label_config: str | None = None  # events carry the project's config; used if not set up


@dataclass(frozen=True)
class IngestOutcome:
    status: Literal["stored", "skipped"]
    reason: str = ""
    box_count: int = 0
    training_set_size: int = 0
    training_launched: bool = False
    training_reason: str = ""  # why no Training Run started, when it did not


def _skipped(reason: str) -> IngestOutcome:
    return IngestOutcome("skipped", reason=reason)


class IngestAnnotation:
    def __init__(
        self,
        setup_project: SetupProject,
        recorder: ExampleRecorder,
        examples: ExampleStore,
        trainer: Trainer,
        policy: RetrainPolicy,
        tracker: ExperimentTracker,
        serving_version: Callable[[], str],
    ) -> None:
        self._setup_project = setup_project
        self._recorder = recorder
        self._examples = examples
        self._trainer = trainer
        self._policy = policy
        self._tracker = tracker
        self._serving_version = serving_version
        self._retrain_lock = threading.Lock()  # webhooks arrive on parallel threads

    def rejection(self, event: AnnotationEvent) -> str | None:
        """Why this event cannot be ingested at all, decided before any work is scheduled."""
        return self._setup_project.not_ready_reason(event.label_config)

    def __call__(self, event: AnnotationEvent) -> IngestOutcome:
        outcome = self._ingest(event)
        if outcome.status != "stored":
            logger.info("task %s: skipped (%s)", event.task.id, outcome.reason)
        elif outcome.training_launched:
            logger.info(
                "task %s: stored an Example with %d boxes; Training Set has %d positive Examples;"
                " Training Run launched",
                event.task.id,
                outcome.box_count,
                outcome.training_set_size,
            )
        else:
            logger.info(
                "task %s: stored an Example with %d boxes; Training Set has %d positive Examples;"
                " no Training Run (%s)",
                event.task.id,
                outcome.box_count,
                outcome.training_set_size,
                outcome.training_reason,
            )
        return outcome

    def _ingest(self, event: AnnotationEvent) -> IngestOutcome:
        rejection = self.rejection(event)
        if rejection:
            return _skipped(rejection)
        context = self._setup_project.current_or_establish(event.label_config)
        assert context is not None  # rejection() guarantees it
        reason = self._recorder.record(event.task, event.annotation, context)
        if reason:
            return _skipped(reason)
        with self._retrain_lock:  # one decision at a time, so a threshold launches one run
            size = self._examples.positive_count()
            decision = self._retrain(size)
        return IngestOutcome(
            "stored",
            box_count=len(event.annotation.regions),
            training_set_size=size,
            training_launched=decision.should_train,
            training_reason=decision.reason,
        )

    def _retrain(self, training_set_size: int) -> RetrainDecision:
        """Apply the retrain rule and act on it; the decision returned is what happened."""
        decision = decide_retrain(
            self._policy,
            training_set_size=training_set_size,
            run_active=lambda: self._trainer.active() is not None,
        )
        if not decision.should_train:
            return decision
        facts = LaunchFacts(
            trigger=RunTrigger.RETRAIN_THRESHOLD,
            training_set_size=training_set_size,
            serving_version=self._serving_version(),
        )
        try:
            self._trainer.start(tracked_as=self._tracker.record_launch(facts))
        except TrainingAlreadyActive as e:
            return RetrainDecision(False, str(e))
        return decision
