"""Turning a Task's Annotation into a stored Example: shared by the webhook and Start Training."""

from lb_anythings.application.ports import ExampleStore, TaskMediaResolver
from lb_anythings.application.project_context import ProjectContext
from lb_anythings.domain.annotation import Annotation, GroundTruthBox
from lb_anythings.domain.annotation_target import AnnotationTarget
from lb_anythings.domain.errors import MediaUnavailable
from lb_anythings.domain.example import Example
from lb_anythings.domain.task import Task


class ExampleRecorder:
    def __init__(self, media: TaskMediaResolver, examples: ExampleStore) -> None:
        self._media = media
        self._examples = examples

    def record(self, task: Task, annotation: Annotation, context: ProjectContext) -> str | None:
        """Save the Example; return the reason when nothing was saved."""
        if annotation.cancelled:
            return "the annotation was cancelled or skipped"
        if task.id is None:
            return "the task has no id"
        reference = task.image_reference(context.target)
        if reference is None:
            return f"the task has no {context.target.image_field!r} field"
        try:
            image = self._media.load(reference, context.credentials)
        except MediaUnavailable as e:
            return str(e)
        boxes = tuple(_ground_truth(region, context.target) for region in annotation.regions)
        self._examples.save(Example(str(task.id), boxes), image)
        return None


def _ground_truth(region: GroundTruthBox, target: AnnotationTarget) -> GroundTruthBox:
    """An Annotator's label is kept as given; only a missing one takes the project's first."""
    if region.label or not target.labels:
        return region
    return GroundTruthBox(region.box, target.labels[0])
