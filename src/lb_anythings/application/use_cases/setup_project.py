"""Set the server up for a Label Studio project."""

from lb_anythings.application.project_context import (
    Credentials,
    ProjectContext,
    ProjectContextHolder,
)
from lb_anythings.domain.annotation_target import parse_label_config


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
            context = self(label_config, Credentials())
        return context
