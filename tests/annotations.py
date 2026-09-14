"""Annotation payload pieces shared by tests, in the shape Label Studio sends them."""

from typing import Any

RECTANGLE: dict[str, Any] = {
    "type": "rectanglelabels",
    "value": {"x": 10.0, "y": 10.0, "width": 50.0, "height": 50.0, "rectanglelabels": ["pipe"]},
}
CHOICE: dict[str, Any] = {"type": "choices", "value": {"choices": ["ok"]}}
