"""An Example: one image with its ground-truth boxes, kept for training. One per Task."""

from dataclasses import dataclass

from lb_anythings.domain.annotation import GroundTruthBox


@dataclass(frozen=True)
class Example:
    task_id: str
    boxes: tuple[GroundTruthBox, ...]

    @property
    def is_positive(self) -> bool:
        """Only Examples with at least one box count toward the Training Set size."""
        return bool(self.boxes)
