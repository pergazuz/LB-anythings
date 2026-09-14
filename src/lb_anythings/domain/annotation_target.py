"""The Annotation Target: what the label config tells us to address."""

from dataclasses import dataclass
from xml.etree import ElementTree

from lb_anythings.domain.errors import InvalidLabelConfig

CONTROL_TAG = "RectangleLabels"
OBJECT_TAG = "Image"


@dataclass(frozen=True)
class AnnotationTarget:
    control_name: str
    object_name: str
    image_field: str
    labels: tuple[str, ...]

    def label_for(self, detected: str) -> str:
        """Map a Detector's own class name onto the project's labels.

        An exact match is kept; anything else lands on the project's first label, so a
        single-class Detector always yields the project's label whatever it calls its class.
        """
        if detected in self.labels or not self.labels:
            return detected
        return self.labels[0]


def parse_label_config(xml: str | None) -> AnnotationTarget:
    if not xml or not xml.strip():
        raise InvalidLabelConfig("label config is empty")
    try:
        root = ElementTree.fromstring(xml)
    except ElementTree.ParseError as e:
        raise InvalidLabelConfig(f"label config is not valid XML: {e}") from e

    controls = [el for el in root.iter(CONTROL_TAG)]
    if len(controls) != 1:
        names = ", ".join(el.get("name", "?") for el in controls)
        found = f"{len(controls)} ({names})" if controls else "0"
        raise InvalidLabelConfig(f"expected exactly one {CONTROL_TAG} control, found {found}")
    control = controls[0]
    control_name = control.get("name") or ""
    to_name = control.get("toName") or ""

    images = list(root.iter(OBJECT_TAG))
    targets = [el for el in images if el.get("name") == to_name]
    if len(targets) != 1:
        found = ", ".join(el.get("name", "?") for el in images) if images else "none"
        raise InvalidLabelConfig(
            f"{CONTROL_TAG} '{control_name}' targets '{to_name}' but expected exactly one "
            f"{OBJECT_TAG} with that name, found {len(targets)} (all {OBJECT_TAG} tags: {found})"
        )
    image_field = (targets[0].get("value") or "").lstrip("$")
    labels = tuple(label.get("value", "") for label in control.iter("Label"))
    return AnnotationTarget(control_name, to_name, image_field, labels)
