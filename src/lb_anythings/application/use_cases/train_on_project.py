"""Train on the whole project: what the Start Training button does."""

import logging
from dataclasses import dataclass, replace

from lb_anythings.application.example_recording import ExampleRecorder
from lb_anythings.application.ports import ExampleStore, LabelStudioProjectClient, Trainer
from lb_anythings.application.project_context import NO_CREDENTIALS, Credentials
from lb_anythings.application.use_cases.setup_project import SetupProject
from lb_anythings.domain.errors import ProjectExportFailed, TrainingAlreadyActive

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class StartTrainingRequest:
    project_id: int | None
    label_config: str | None = None  # the event carries the project's config; used if not set up


@dataclass(frozen=True)
class TrainOnProjectOutcome:
    collected: int = 0
    unannotated: int = 0
    not_collected: int = 0
    training_launched: bool = False
    reason: str = ""  # why no Training Run started, or why nothing happened at all


class TrainOnProject:
    def __init__(
        self,
        setup_project: SetupProject,
        project_client: LabelStudioProjectClient,
        recorder: ExampleRecorder,
        examples: ExampleStore,
        trainer: Trainer,
        *,
        minimum_examples: int,
        configured_credentials: Credentials,
    ) -> None:
        self._setup_project = setup_project
        self._project_client = project_client
        self._recorder = recorder
        self._examples = examples
        self._trainer = trainer
        self._minimum = minimum_examples
        self._configured = configured_credentials

    def credentials(self) -> Credentials:
        """Setup's credentials, filled in from the configured ones; they win as a pair."""
        context = self._setup_project.current()
        given = context.credentials if context else NO_CREDENTIALS
        return given.resolved_against(self._configured)

    def rejection(self, request: StartTrainingRequest) -> str | None:
        """Why the project cannot be pulled at all, decided before any work is scheduled."""
        not_ready = self._setup_project.not_ready_reason(request.label_config)
        if not_ready:
            return not_ready
        credentials = self.credentials()
        missing = []
        if not credentials.hostname:
            missing.append("LABEL_STUDIO_URL")
        if not credentials.access_token:
            missing.append("LABEL_STUDIO_API_KEY")
        if missing:
            return (
                f"cannot pull the project: set {' and '.join(missing)} (Label Studio sends a "
                "hostname and access token at setup, which are used when present)"
            )
        if request.project_id is None:
            return "cannot pull the project: the request names no project id"
        return None

    def __call__(self, request: StartTrainingRequest) -> TrainOnProjectOutcome:
        outcome = self._train(request)
        logger.info(
            "Start Training: collected %d Examples (%d unannotated, %d unreadable); %s",
            outcome.collected,
            outcome.unannotated,
            outcome.not_collected,
            "Training Run launched"
            if outcome.training_launched
            else f"no Training Run ({outcome.reason})",
        )
        return outcome

    def _train(self, request: StartTrainingRequest) -> TrainOnProjectOutcome:
        rejection = self.rejection(request)
        if rejection:
            return TrainOnProjectOutcome(reason=rejection)
        context = self._setup_project.current_or_establish(request.label_config)
        assert context is not None and request.project_id is not None  # rejection() guarantees

        try:
            exported = self._project_client.exported_tasks(request.project_id, self.credentials())
        except ProjectExportFailed as e:
            return TrainOnProjectOutcome(reason=str(e))

        collected = unannotated = not_collected = 0
        for item in exported:
            if item.annotation is None:
                unannotated += 1
                continue
            reason = self._recorder.record(item.task, item.annotation, context)
            if reason:
                logger.info("task %s: not collected (%s)", item.task.id, reason)
                not_collected += 1
            else:
                collected += 1

        counted = TrainOnProjectOutcome(collected, unannotated, not_collected)
        size = self._examples.positive_count()
        if size < self._minimum:
            reason = f"Training Set has {size}; need at least {self._minimum}"
            return replace(counted, reason=reason)
        try:
            self._trainer.start()
        except TrainingAlreadyActive as e:
            return replace(counted, reason=str(e))
        return replace(counted, training_launched=True)
