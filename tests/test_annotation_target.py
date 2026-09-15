"""A label config and the Annotation Target it describes: read, and written (pure domain)."""

import re

import pytest

from lb_anythings.domain.annotation_target import (
    AnnotationTarget,
    labeling_config,
    parse_label_config,
)
from lb_anythings.domain.errors import InvalidLabelConfig
from tests.label_configs import PIPE, VEHICLES

NO_CONTROL = '<View><Image name="image" value="$image"/><Choices name="c" toName="image"/></View>'

TWO_CONTROLS = """
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="a" toName="image"><Label value="x"/></RectangleLabels>
  <RectangleLabels name="b" toName="image"><Label value="y"/></RectangleLabels>
</View>
"""

WRONG_TARGET = """
<View>
  <Image name="image" value="$image"/>
  <RectangleLabels name="label" toName="picture"><Label value="pipe"/></RectangleLabels>
</View>
"""

TWO_IMAGES_SAME_NAME = """
<View>
  <Image name="image" value="$a"/>
  <Image name="image" value="$b"/>
  <RectangleLabels name="label" toName="image"><Label value="pipe"/></RectangleLabels>
</View>
"""


def test_reads_control_object_field_and_labels() -> None:
    assert parse_label_config(PIPE) == AnnotationTarget(
        control_name="label", object_name="image", image_field="image", labels=("pipe",)
    )


def test_any_tag_names_work_and_the_dollar_is_stripped_from_the_field() -> None:
    target = parse_label_config(VEHICLES)

    assert (target.control_name, target.object_name, target.image_field) == (
        "boxes",
        "img",
        "photo",
    )
    assert target.labels == ("car", "truck")


@pytest.mark.parametrize("xml", ["", "   ", None])
def test_rejects_an_empty_config(xml: str | None) -> None:
    with pytest.raises(InvalidLabelConfig, match="label config is empty"):
        parse_label_config(xml)


def test_rejects_text_that_is_not_xml() -> None:
    with pytest.raises(InvalidLabelConfig, match="not valid XML"):
        parse_label_config("<View><Image name='x'></View>")


def test_rejects_a_config_with_no_rectangle_labels() -> None:
    with pytest.raises(InvalidLabelConfig, match="exactly one RectangleLabels control, found 0"):
        parse_label_config(NO_CONTROL)


def test_rejects_a_config_with_two_rectangle_labels_naming_both() -> None:
    with pytest.raises(InvalidLabelConfig, match=r"found 2 \(a, b\)"):
        parse_label_config(TWO_CONTROLS)


def test_rejects_a_control_whose_target_is_not_an_image() -> None:
    with pytest.raises(InvalidLabelConfig, match="targets 'picture'.*found 0.*image"):
        parse_label_config(WRONG_TARGET)


def test_rejects_two_images_with_the_targeted_name() -> None:
    with pytest.raises(InvalidLabelConfig, match="expected exactly one Image.*found 2"):
        parse_label_config(TWO_IMAGES_SAME_NAME)


# --- building a label config for a project that does not exist yet ---


def test_a_built_config_parses_back_to_the_labels_it_was_built_from() -> None:
    """The builder is the inverse of the parser: whatever it writes, the backend can read."""
    assert parse_label_config(labeling_config(["pipe"])) == AnnotationTarget(
        control_name="label", object_name="image", image_field="image", labels=("pipe",)
    )


def test_a_built_config_keeps_the_order_of_several_labels() -> None:
    target = parse_label_config(labeling_config(["car", "truck", "van"]))

    assert target.labels == ("car", "truck", "van")


def test_every_label_gets_a_colour_of_its_own() -> None:
    """Boxes of different classes are told apart by colour while annotating."""
    config = labeling_config(["car", "truck", "van"])

    colours = re.findall(r'background="([^"]+)"', config)
    assert len(colours) == 3 and len(set(colours)) == 3


def test_a_label_that_would_break_the_xml_survives_the_round_trip() -> None:
    target = parse_label_config(labeling_config(['pipe & "steel" <90mm>']))

    assert target.labels == ('pipe & "steel" <90mm>',)


def test_the_image_field_can_be_named() -> None:
    """Whatever key the imported Tasks carry the image under is the one to read."""
    target = parse_label_config(labeling_config(["pipe"], image_field="frame"))

    assert target.image_field == "frame"


def test_a_project_needs_at_least_one_label() -> None:
    with pytest.raises(InvalidLabelConfig, match="at least one label"):
        labeling_config([])


def test_a_blank_label_is_refused() -> None:
    with pytest.raises(InvalidLabelConfig, match="blank"):
        labeling_config(["pipe", "  "])


def test_the_same_label_twice_is_refused() -> None:
    """Two identical labels are indistinguishable to an Annotator and to the Detector."""
    with pytest.raises(InvalidLabelConfig, match="twice"):
        labeling_config(["pipe", "pipe"])
