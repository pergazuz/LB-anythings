"""Parsing a label config into an Annotation Target (pure domain seam)."""

import pytest

from lb_anythings.domain.annotation_target import AnnotationTarget, parse_label_config
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
