"""Typed errors the core raises. Adapters translate them at the edge."""


class InvalidLabelConfig(ValueError):
    """The label config does not describe exactly one RectangleLabels on one Image."""
