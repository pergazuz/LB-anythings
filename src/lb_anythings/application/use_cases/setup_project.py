"""Set the server up for a Label Studio project."""

from lb_anythings.application.project_context import (
    NO_CREDENTIALS,
    Credentials,
    ProjectContext,
    ProjectContextHolder,
)
from lb_anythings.domain.annotation_target import parse_label_config
from lb_anythings.domain.errors import InvalidLabelConfig

NOT_SET_UP = "the backend is not set up for a project yet; connect it in Label Studio first"


class SetupProject:
    def __init__(self, context_holder: ProjectContextHolder) -> None:
        self._context_holder = context_holder

    def __call__(self, label_config: str | None, credentials: Credentials) -> ProjectContext:
        context = ProjectContext(parse_label_config(label_config), credentials)
        self._context_holder.set(context)
        return context

    def current(self) -> ProjectContext | None:
        return self._context_holder.get()

    def current_or_establish(self, label_config: str | None) -> ProjectContext | None:
        """The current context; or, when there is none and a request carries a label config,
        the context established from it (Label Studio sends it on predict and on events)."""
        context = self._context_holder.get()
        if context is None and label_config:
            context = self(label_config, NO_CREDENTIALS)
        return context

    def not_ready_reason(self, label_config: str | None) -> str | None:
        """None when a request can proceed; otherwise why no project context can be had.

        A request that carries the project's label config establishes the context itself, so
        only an absent or unusable config stops it.
        """
        if self._context_holder.get() is not None:
            return None
        if not label_config:
            return NOT_SET_UP
        try:
            parse_label_config(label_config)
        except InvalidLabelConfig as e:
            return f"the event's label config is unusable: {e}"
        return None
