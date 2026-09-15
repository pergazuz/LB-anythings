"""Typed errors the core raises. Adapters translate them at the edge."""


class InvalidLabelConfig(ValueError):
    """The label config does not describe exactly one RectangleLabels on one Image."""


class MediaUnavailable(RuntimeError):
    """A Task's image could not be fetched or decoded."""


class TrainingAlreadyActive(RuntimeError):
    """A Training Run is running; a second one would fight it for the GPU and the Checkpoint."""


class NotEnoughExamples(ValueError):
    """The Training Set is too small to split into training and validation."""


class ProjectExportFailed(RuntimeError):
    """Label Studio did not hand over the project's tasks."""


class NoCheckpointAvailable(RuntimeError):
    """There is no Checkpoint to work from: nothing has been trained and none is configured."""


class ServiceDidNotStart(RuntimeError):
    """A Service of the Stack never answered, or stopped before it did."""


class ProjectWiringFailed(RuntimeError):
    """Label Studio would not set the project up the way the loop needs it."""
