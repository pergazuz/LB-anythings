"""Boxes: one canonical form, with the Label Studio and YOLO forms as pure conversions."""

import pytest

from lb_anythings.domain.geometry import Box

BOX = Box(0.1, 0.1, 0.6, 0.6)


def test_a_box_is_built_from_pixels_relative_to_its_image() -> None:
    assert Box.from_pixels(20, 10, 120, 60, width=200, height=100) == BOX


def test_label_studio_percent_form_is_top_left_plus_size_in_percent() -> None:
    assert BOX.to_label_studio() == pytest.approx(
        {"x": 10.0, "y": 10.0, "width": 50.0, "height": 50.0}
    )


def test_label_studio_percent_form_is_read_back() -> None:
    box = Box.from_label_studio(x=25.0, y=40.0, width=10.0, height=20.0)

    assert box.coordinates() == pytest.approx((0.25, 0.4, 0.35, 0.6))


def test_yolo_form_is_centre_plus_size_normalized() -> None:
    assert BOX.to_yolo() == pytest.approx((0.35, 0.35, 0.5, 0.5))


def test_yolo_form_is_read_back() -> None:
    assert Box.from_yolo(0.35, 0.35, 0.5, 0.5).coordinates() == pytest.approx(BOX.coordinates())


@pytest.mark.parametrize("box", [BOX, Box(0.0, 0.0, 1.0, 1.0), Box(0.123, 0.456, 0.789, 0.9)])
def test_both_foreign_forms_round_trip(box: Box) -> None:
    via_label_studio = Box.from_label_studio(**box.to_label_studio())
    via_yolo = Box.from_yolo(*box.to_yolo())

    assert via_label_studio.coordinates() == pytest.approx(box.coordinates())
    assert via_yolo.coordinates() == pytest.approx(box.coordinates())


def test_a_box_must_be_ordered_and_inside_the_unit_square() -> None:
    with pytest.raises(ValueError):
        Box(0.6, 0.1, 0.1, 0.6)
    with pytest.raises(ValueError):
        Box(0.0, 0.0, 1.2, 0.5)
