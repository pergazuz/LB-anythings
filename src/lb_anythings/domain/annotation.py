"""An Annotation: the human-corrected regions for one Task, as Label Studio sends them."""

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from lb_anythings.domain.geometry import Box

RECTANGLE_TYPE = "rectanglelabels"


@dataclass(frozen=True)
class GroundTruthBox:
    """A box an Annotator confirmed, with the label they gave it ("" when they gave none)."""

    box: Box
    label: str


@dataclass(frozen=True)
class Annotation:
    regions: tuple[GroundTruthBox, ...]
    cancelled: bool = False


def parse_annotation(payload: Mapping[str, Any]) -> Annotation:
    """Read Label Studio's annotation object. Only rectangle regions matter here."""
    regions: list[GroundTruthBox] = []
    for region in payload.get("result") or []:
        if region.get("type") != RECTANGLE_TYPE:
            continue
        value = region.get("value") or {}
        try:
            box = Box.from_label_studio(
                x=float(value["x"]),
                y=float(value["y"]),
                width=float(value["width"]),
                height=float(value["height"]),
            )
        except (KeyError, TypeError, ValueError):
            continue
        labels = value.get(RECTANGLE_TYPE) or []
        regions.append(GroundTruthBox(box, str(labels[0]) if labels else ""))
    cancelled = bool(payload.get("was_cancelled") or payload.get("skipped"))
    return Annotation(tuple(regions), cancelled)
