"""Composition for `up`: which Services to run, how to start each, and which project.

The rules about *when* a Service is started and how the project is wired live in the use case.
What lives here is everything that is specific to these three programs -- Label Studio's flags,
MLflow's, our own -- because that is knowledge about the world, not about the loop.
"""

import logging
import shlex
import shutil
import sys
import time
import webbrowser
from collections.abc import Callable
from dataclasses import dataclass
from urllib.parse import urlparse

from lb_anythings.adapters.outbound.filesystem.examples import FilesystemExampleStore
from lb_anythings.adapters.outbound.http.probe import HttpHealthProbe
from lb_anythings.adapters.outbound.labelstudio.admin import LabelStudioAdminClient
from lb_anythings.adapters.outbound.labelstudio.auth import looks_like_a_jwt
from lb_anythings.adapters.outbound.subprocess.services import LocalServiceLauncher
from lb_anythings.application.ports import ServiceCommand
from lb_anythings.application.use_cases.bring_up_stack import (
    BringUpStack,
    ProjectPlan,
    RunningService,
    Stack,
    StackPlan,
    shut_down,
)
from lb_anythings.bootstrap.container import configured_credentials, spawn_environment
from lb_anythings.bootstrap.settings import Settings
from lb_anythings.domain.errors import ProjectWiringFailed, ServiceDidNotStart

logger = logging.getLogger(__name__)

LABEL_STUDIO_EXECUTABLE = "label-studio"
WATCH_INTERVAL_SECONDS = 1.0
CANCELLED = 130  # the conventional exit code for "the Operator interrupted it"
DEFAULT_LABEL = "object"  # a project with no Training Set and no --label has to say something


@dataclass(frozen=True)
class StackOptions:
    """What the Operator asked `up` for, over and above the settings."""

    project_id: int | None = None
    title: str | None = None
    always_create: bool = False
    labels: tuple[str, ...] = ()
    label_studio: bool = True
    mlflow: bool = True
    wire: bool = True
    host: str | None = None
    port: int | None = None
    quiet: bool = False
    open_browser: bool = True


def stack_plan(settings: Settings, options: StackOptions) -> StackPlan:
    services = [
        service
        for service in (
            label_studio_service(settings) if options.label_studio else None,
            mlflow_service(settings) if options.mlflow else None,
            backend_service(settings, options),
        )
        if service is not None
    ]
    return StackPlan(
        services=tuple(services),
        project=project_plan(settings, options) if options.wire else None,
        ready_timeout=settings.stack_timeout,
    )


def label_studio_service(settings: Settings) -> ServiceCommand:
    base = settings.label_studio_base
    command = _label_studio_command(settings)
    if settings.label_studio_command is None:  # our flags only fit our own invocation
        command += ("start", "--no-browser", "--port", str(_port_of(base, 8080)))
        if settings.label_studio_data_dir is not None:
            command += ("--data-dir", str(settings.label_studio_data_dir))
    return ServiceCommand(
        name="label-studio",
        command=command,
        url=base,
        environment=_first_user(settings),
    )


def _label_studio_command(settings: Settings) -> tuple[str, ...]:
    if settings.label_studio_command:
        return tuple(part.strip('"') for part in shlex.split(settings.label_studio_command))
    found = shutil.which(LABEL_STUDIO_EXECUTABLE)
    if found is None:
        raise ServiceDidNotStart(
            "Label Studio is not installed here: `uv tool install label-studio` (or "
            "`pip install label-studio`), or set LB_LABEL_STUDIO_COMMAND to how you start it, "
            "or run `up --no-label-studio` beside one you start yourself"
        )
    return (found,)


def _first_user(settings: Settings) -> dict[str, str]:
    """Credentials for a Label Studio with no user yet. It ignores them once one exists.

    Through the environment rather than the command line, which every process list can read.
    """
    password = settings.label_studio_password
    if not settings.label_studio_username or password is None:
        return {}
    environment = {
        "LABEL_STUDIO_USERNAME": settings.label_studio_username,
        "LABEL_STUDIO_PASSWORD": password.get_secret_value(),
    }
    key = settings.label_studio_api_key
    # This sets a *legacy* token, so it only makes sense for one -- and legacy tokens are off
    # by default in 1.23, hence the second variable. A personal access token is issued by
    # Label Studio itself and cannot be planted this way.
    if key is not None and not looks_like_a_jwt(key.get_secret_value()):
        environment["LABEL_STUDIO_USER_TOKEN"] = key.get_secret_value()
        environment["LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN"] = "true"
    return environment


def mlflow_service(settings: Settings) -> ServiceCommand | None:
    """The MLflow UI over the same store the Training Runs write to, or None when there is none.

    A tracking URI that is already a server is someone else's UI: nothing to serve here.
    """
    if not settings.tracking or _is_remote(settings.tracking_uri):
        return None
    url = settings.tracking_ui_url.rstrip("/")
    return ServiceCommand(
        name="mlflow",
        command=(
            sys.executable,
            "-m",
            "mlflow",
            "ui",
            "--backend-store-uri",
            tracking_store_uri(settings),
            "--default-artifact-root",
            str(settings.tracking_dir / "artifacts"),
            "--host",
            urlparse(url).hostname or "127.0.0.1",
            "--port",
            str(_port_of(url, 5000)),
        ),
        url=url,
    )


def tracking_store_uri(settings: Settings) -> str:
    """Where recorded Training Runs live: the configured URI, or SQLite in the data directory."""
    return settings.tracking_uri or f"sqlite:///{(settings.tracking_dir / 'mlflow.db').as_posix()}"


def backend_service(settings: Settings, options: StackOptions) -> ServiceCommand:
    host = options.host or settings.host
    port = options.port or settings.port
    return ServiceCommand(
        name="backend",
        command=(
            sys.executable,
            "-m",
            "lb_anythings",
            "serve",
            "--host",
            host,
            "--port",
            str(port),
        ),
        # Not the bind address: 0.0.0.0 is where it listens, not somewhere to ask.
        url=f"http://127.0.0.1:{port}",
        environment=spawn_environment(settings),
        # A Training Run is the backend's child and outlives it on purpose (ADR 0002):
        # stopping the Stack mid-run must not throw away the run.
        stop_descendants=False,
    )


def project_plan(settings: Settings, options: StackOptions) -> ProjectPlan | None:
    """Which project to wire, or None when there is no way to ask Label Studio anything."""
    if settings.label_studio_api_key is None:
        return None
    port = options.port or settings.port
    reachable_at = settings.backend_url or f"http://localhost:{port}"
    return ProjectPlan(
        title=options.title or settings.project_title,
        model_url=reachable_at.rstrip("/"),
        project_id=options.project_id if options.project_id is not None else settings.project,
        always_create=options.always_create,
        labels=options.labels or _labels(settings),
    )


def _labels(settings: Settings) -> tuple[str, ...]:
    """What a created project should offer: whatever the Training Set is already labelled with."""
    collected = tuple(FilesystemExampleStore(settings.examples_dir).class_names())
    return collected or (DEFAULT_LABEL,)


def stack_bringer(settings: Settings, echo: Callable[[str], None] | None = None) -> BringUpStack:
    return BringUpStack(
        LocalServiceLauncher(settings.logs_dir, echo),
        HttpHealthProbe(),
        LabelStudioAdminClient(configured_credentials(settings)),
    )


def project_page(settings: Settings, project_id: int) -> str:
    return f"{settings.label_studio_base}/projects/{project_id}/data"


def _is_remote(uri: str | None) -> bool:
    return bool(uri) and str(uri).startswith(("http://", "https://"))


def _port_of(url: str, fallback: int) -> int:
    parsed = urlparse(url)
    return parsed.port or (443 if parsed.scheme == "https" else fallback)


def run_stack(
    settings: Settings,
    options: StackOptions,
    *,
    bring_up: Callable[[StackPlan], Stack] | None = None,
    wait: Callable[[Stack], str | None] = lambda stack: watch(stack),
    out: Callable[[str], None] = print,
) -> int:
    """`lb-anythings up`: bring the Stack up, say where everything is, and hold it there."""
    bring_up = bring_up or stack_bringer(settings, echo=None if options.quiet else out)
    try:
        stack = bring_up(stack_plan(settings, options))
    except ServiceDidNotStart as e:
        print(f"cannot bring the Stack up: {e}", file=sys.stderr)
        print(f"what each Service printed is in {settings.logs_dir}", file=sys.stderr)
        return 1
    except ProjectWiringFailed as e:
        print(f"cannot bring the Stack up: {e}", file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        # Label Studio takes half a minute to start, which is long enough to change your mind.
        # Whatever had been started has already been stopped on the way out.
        out("")
        out("stopped before the Stack was up")
        return CANCELLED

    for line in summary(settings, options, stack):
        out(line)
    if options.open_browser and stack.project is not None:
        webbrowser.open(project_page(settings, stack.project.id))

    stopped = None
    try:
        stopped = wait(stack)
        if stopped is not None:
            print(f"{stopped} stopped on its own; see {settings.logs_dir}", file=sys.stderr)
    except KeyboardInterrupt:
        out("")  # the ^C the terminal just echoed
    finally:
        out("stopping the Stack")
        shut_down(stack)
    return 1 if stopped is not None else 0


def watch(stack: Stack, interval: float = WATCH_INTERVAL_SECONDS) -> str | None:
    """Block until a Service that `up` started stops; the name of the first one that does."""
    while True:
        for service in stack.started:
            if service.process is not None and not service.process.running():
                return service.name
        time.sleep(interval)


def summary(settings: Settings, options: StackOptions, stack: Stack) -> list[str]:
    """The one block an Operator reads: where everything is, and what is still to do."""
    lines = [""]
    for service in stack.services:
        lines.append(f"  {service.name:<13} {service.url:<32} {_how(service)}")
    lines.append(f"  {'logs':<13} {str(settings.logs_dir):<32}")
    lines.append("")
    lines.extend(_project_lines(settings, options, stack))
    lines.append("")
    lines.append("Ctrl+C stops what `up` started; anything already running is left alone.")
    return lines


def _how(service: RunningService) -> str:
    return "already running" if service.adopted else "started"


def _project_lines(settings: Settings, options: StackOptions, stack: Stack) -> list[str]:
    project = stack.project
    if project is not None:
        state = "created" if project.created else "existing"
        model = "model connected" if project.model_connected else "model already connected"
        lines = [
            f"  project: {project.title} (#{project.id}, {state}, {model})",
            f"  open it: {project_page(settings, project.id)}",
        ]
        if not project.predictions_while_labelling:
            # Everything else can be right and still no box ever appears on a Task.
            lines += [
                "  note: interactive preannotations are off for this model, so Label",
                "        Studio never asks it for a Prediction while you label. Turn it",
                "        on under Settings -> Model -> Edit.",
            ]
        return lines
    if not options.wire:
        return ["  no project was touched (--no-wiring)."]
    return [
        "  no project was wired: there is no Label Studio token to do it with.",
        f"  Sign in at {settings.label_studio_base}, copy Account & Settings -> Access Token",
        "  into LABEL_STUDIO_API_KEY in your .env, and run `up` again.",
    ]
