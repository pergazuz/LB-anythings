"""Predict for Tasks: what Label Studio asks for when an Annotator opens one."""

import logging
from collections.abc import Sequence
from dataclasses import dataclass

from lb_anythings.application.detector_cache import DetectorCache
from lb_anythings.application.ports import Detector, TaskMediaResolver
from lb_anythings.application.project_context import (
    Credentials,
    ProjectContext,
    ProjectContextHolder,
)
from lb_anythings.application.use_cases.setup_project import SetupProject
from lb_anythings.domain.annotation_target import AnnotationTarget
from lb_anythings.domain.detection import Detection
from lb_anythings.domain.errors import MediaUnavailable
from lb_anythings.domain.prediction import Prediction
from lb_anythings.domain.task import Task

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PredictionBatch:
    """One Prediction per Task, in order, plus what the HTTP adapter needs to address them."""

    target: AnnotationTarget | None
    predictions: tuple[Prediction, ...]
    version: str | None


class PredictTasks:
    def __init__(
        self,
        context_holder: ProjectContextHolder,
        setup_project: SetupProject,
        detector_cache: DetectorCache,
        media: TaskMediaResolver,
    ) -> None:
        self._context_holder = context_holder
        self._setup_project = setup_project
        self._detector_cache = detector_cache
        self._media = media

    def __call__(
        self,
        tasks: Sequence[Task],
        label_config: str | None = None,
        force_reload: bool = False,
    ) -> PredictionBatch:
        context = self._context_holder.get()
        if context is None:
            if not label_config:
                return PredictionBatch(None, (), None)
            context = self._setup_project(label_config, Credentials())

        detector = self._detector_cache.current(force_reload=force_reload)
        predictions = tuple(self._predict_one(task, context, detector) for task in tasks)
        return PredictionBatch(context.target, predictions, detector.version)

    def _predict_one(self, task: Task, context: ProjectContext, detector: Detector) -> Prediction:
        empty = Prediction((), detector.version)
        reference = task.image_reference(context.target)
        if reference is None:
            logger.warning(
                "task %s has no %r field; empty Prediction", task.id, context.target.image_field
            )
            return empty
        try:
            image = self._media.load(reference, context.credentials)
        except MediaUnavailable as e:
            logger.warning("task %s: %s; empty Prediction", task.id, e)
            return empty
        regions = tuple(
            Detection(d.box, d.score, context.target.label_for(d.label))
            for d in detector.detect(image)
        )
        return Prediction(regions, detector.version)
