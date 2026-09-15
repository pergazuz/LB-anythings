"""Bring the Stack up: Label Studio, the tracking UI and this backend, wired to a project.

Doing it by hand is four things in three places and one toggle that fails silently when it is
off. This is that, once, in an order that holds: nothing is wired until every Service answers,
because Label Studio health-checks a model before it will accept one.
"""

import logging
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from lb_anythings.application.ports import (
    HealthProbe,
    LabelStudioProject,
    LabelStudioProjectAdmin,
    ServiceCommand,
    ServiceLauncher,
    ServiceProcess,
)
from lb_anythings.domain.annotation_target import labeling_config
from lb_anythings.domain.errors import ProjectWiringFailed, ServiceDidNotStart

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ProjectPlan:
    """Which Label Studio project the Stack is for, and what to do if it is not there yet."""

    title: str  # found or created by this, unless an id says otherwise
    model_url: str  # this backend, as Label Studio must reach it
    project_id: int | None = None  # use exactly this project
    always_create: bool = False  # create one even if the title is taken
    labels: tuple[str, ...] = ()  # what a created project's config offers
    model_title: str = "lb-anythings"


@dataclass(frozen=True)
class StackPlan:
    services: tuple[ServiceCommand, ...] = ()
    project: ProjectPlan | None = None  # None: bring the Services up and wire nothing
    ready_timeout: float = 180.0  # a first Label Studio start runs its migrations
    poll_seconds: float = 0.5


@dataclass(frozen=True)
class RunningService:
    name: str
    url: str
    adopted: bool  # it was already answering, so it is not ours to stop
    process: ServiceProcess | None = None


@dataclass(frozen=True)
class WiredProject:
    id: int
    title: str
    created: bool
    model_connected: bool  # False when it was already connected
    predictions_while_labelling: bool = True  # one connected by hand may have it off


@dataclass(frozen=True)
class Stack:
    services: tuple[RunningService, ...] = ()
    project: WiredProject | None = None
    started: tuple[RunningService, ...] = field(default_factory=tuple)


def shut_down(stack: Stack) -> None:
    """Stop what was started, newest first, and leave what was adopted running."""
    for service in reversed(stack.started):
        if service.process is None:
            continue
        logger.info("stopping %s", service.name)
        try:
            service.process.stop()
        except OSError as e:  # one Service refusing to die must not strand the others
            logger.warning("could not stop %s: %s", service.name, e)


class BringUpStack:
    def __init__(
        self,
        launcher: ServiceLauncher,
        probe: HealthProbe,
        admin: LabelStudioProjectAdmin,
        *,
        sleep: Callable[[float], None] = time.sleep,
        now: Callable[[], float] = time.monotonic,
    ) -> None:
        self._launcher = launcher
        self._probe = probe
        self._admin = admin
        self._sleep = sleep
        self._now = now

    def __call__(self, plan: StackPlan) -> Stack:
        services = tuple(self._start(service) for service in plan.services)
        started = tuple(service for service in services if not service.adopted)
        stack = Stack(services, None, started)
        try:
            self._wait_for(plan, services)
            project = self._wire(plan.project) if plan.project else None
        except BaseException:
            shut_down(stack)  # a half-started Stack is worse than none: the next `up` adopts it
            raise
        return Stack(services, project, started)

    def _start(self, service: ServiceCommand) -> RunningService:
        if self._probe.answers(service.health_url):
            logger.info("%s is already running at %s", service.name, service.url)
            return RunningService(service.name, service.url, adopted=True)
        logger.info("starting %s", service.name)
        return RunningService(
            service.name, service.url, adopted=False, process=self._launcher.launch(service)
        )

    def _wait_for(self, plan: StackPlan, services: Sequence[RunningService]) -> None:
        deadline = self._now() + plan.ready_timeout
        for command, service in zip(plan.services, services, strict=True):
            if service.adopted:
                continue
            self._wait_for_one(command, service, deadline, plan.poll_seconds)
            logger.info("%s is up at %s", service.name, service.url)

    def _wait_for_one(
        self, command: ServiceCommand, service: RunningService, deadline: float, poll: float
    ) -> None:
        while not self._probe.answers(command.health_url):
            if service.process is not None and not service.process.running():
                raise ServiceDidNotStart(
                    f"{service.name} stopped before it answered; its log says why"
                )
            if self._now() >= deadline:
                raise ServiceDidNotStart(
                    f"{service.name} did not answer at {command.health_url} in time"
                )
            self._sleep(poll)

    def _wire(self, plan: ProjectPlan) -> WiredProject:
        project, created = self._project(plan)
        self._admin.enable_training_on_submit(project.id)
        connected, interactive = self._connect_model(project.id, plan)
        logger.info("project %s (%s) is wired to %s", project.id, project.title, plan.model_url)
        return WiredProject(project.id, project.title, created, connected, interactive)

    def _project(self, plan: ProjectPlan) -> tuple[LabelStudioProject, bool]:
        if plan.project_id is not None:
            existing = self._admin.project(plan.project_id)
            if existing is None:
                raise ProjectWiringFailed(f"no Label Studio project {plan.project_id}")
            return existing, False
        if not plan.always_create:
            existing = self._admin.project_titled(plan.title)
            if existing is not None:
                return existing, False
        return self._admin.create_project(plan.title, labeling_config(plan.labels)), True

    def _connect_model(self, project_id: int, plan: ProjectPlan) -> tuple[bool, bool]:
        """Connect this backend if it is not already, and say whether Label Studio
        will ask it for Predictions while a Task is open.

        A connection made by hand may have interactive Predictions off, and then no
        box ever appears. Changing it is the Operator's call, so this only reports it.
        """
        wanted = plan.model_url.rstrip("/")
        connected = self._admin.connected_models(project_id)
        already = next((m for m in connected if m.url.rstrip("/") == wanted), None)
        if already is not None:
            logger.info("%s is already connected to project %s", plan.model_url, project_id)
            return False, already.interactive
        self._admin.connect_model(project_id, plan.model_url, plan.model_title)
        return True, True
