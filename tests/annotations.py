"""Annotation payload pieces shared by tests, in the shape Label Studio sends them."""

from typing import Any

RECTANGLE: dict[str, Any] = {
    "type": "rectanglelabels",
    "value": {"x": 10.0, "y": 10.0, "width": 50.0, "height": 50.0, "rectanglelabels": ["pipe"]},
}
CHOICE: dict[str, Any] = {"type": "choices", "value": {"choices": ["ok"]}}


def annotation_event(
    action: str = "ANNOTATION_CREATED",
    *,
    task_id: int = 7,
    image: str = "a.jpg",
    image_field: str = "image",
    result: list[dict[str, Any]] | None = None,
    **annotation_fields: Any,
) -> dict[str, Any]:
    """A Label Studio annotation webhook payload with one rectangle unless told otherwise."""
    return {
        "action": action,
        "task": {"id": task_id, "data": {image_field: image}},
        "annotation": {"result": [RECTANGLE] if result is None else result, **annotation_fields},
        "project": {"id": 1},
    }
