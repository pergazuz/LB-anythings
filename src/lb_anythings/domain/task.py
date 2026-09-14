"""A Task: one Label Studio task, one image to annotate."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from lb_anythings.domain.annotation_target import AnnotationTarget


@dataclass(frozen=True)
class Task:
    id: int | str | None
    data: Mapping[str, Any] = field(default_factory=dict)

    def image_reference(self, target: AnnotationTarget) -> str | None:
        """The image reference in the field the Annotation Target names, if the Task has one."""
        value = self.data.get(target.image_field)
        return value if isinstance(value, str) and value else None
