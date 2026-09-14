"""Ingest an Annotation: what an Annotator's submit or update turns into."""

import logging
import threading
from dataclasses import dataclass
from typing import Literal

from lb_anythings.application.ports import ExampleStore, TaskMediaResolver, Trainer
from lb_anythings.application.use_cases.setup_project import SetupProject
from lb_anythings.domain.annotation import Annotation, GroundTruthBox
from lb_anythings.domain.annotation_target import AnnotationTarget, parse_label_config
from lb_anythings.domain.errors import InvalidLabelConfig, MediaUnavailable, TrainingAlreadyActive
from lb_anythings.domain.example import Example
from lb_anythings.domain.retraining import RetrainDecision, RetrainPolicy, decide_retrain
from lb_anythings.domain.task import Task

logger = logging.getLogger(__name__)

NOT_SET_UP = "the backend is not set up for a project yet; connect it in Label Studio first"


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
        media: TaskMediaResolver,
        examples: ExampleStore,
        trainer: Trainer,
        policy: RetrainPolicy,
    ) -> None:
        self._setup_project = setup_project
        self._media = media
        self._examples = examples
        self._trainer = trainer
        self._policy = policy
        self._retrain_lock = threading.Lock()  # webhooks arrive on parallel threads

    def rejection(self, event: AnnotationEvent) -> str | None:
        """Why this event cannot be ingested at all, decided before any work is scheduled."""
        if self._setup_project.current() is not None:
            return None
        if not event.label_config:
            return NOT_SET_UP
        try:
            parse_label_config(event.label_config)
        except InvalidLabelConfig as e:
            return f"the event's label config is unusable: {e}"
        return None

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
        if event.annotation.cancelled:
            return _skipped("the annotation was cancelled or skipped")
        if event.task.id is None:
            return _skipped("the event names no task")
        reference = event.task.image_reference(context.target)
        if reference is None:
            return _skipped(f"the task has no {context.target.image_field!r} field")
        try:
            image = self._media.load(reference, context.credentials)
        except MediaUnavailable as e:
            return _skipped(str(e))

        boxes = tuple(_ground_truth(region, context.target) for region in event.annotation.regions)
        self._examples.save(Example(str(event.task.id), boxes), image)
        with self._retrain_lock:  # one decision at a time, so a threshold launches one run
            size = self._examples.positive_count()
            decision = self._retrain(size)
        return IngestOutcome(
            "stored",
            box_count=len(boxes),
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
        try:
            self._trainer.start()
        except TrainingAlreadyActive as e:
            return RetrainDecision(False, str(e))
        return decision


def _ground_truth(region: GroundTruthBox, target: AnnotationTarget) -> GroundTruthBox:
    """An Annotator's label is kept as given; only a missing one takes the project's first."""
    if region.label or not target.labels:
        return region
    return GroundTruthBox(region.box, target.labels[0])
