"""What `up` decides to run, before it runs anything: the plan the composition root builds."""

import shutil
import sys
from pathlib import Path

import pytest
from pydantic import SecretStr

from lb_anythings.application.use_cases.bring_up_stack import StackPlan
from lb_anythings.bootstrap.settings import Settings
from lb_anythings.bootstrap.stack import StackOptions, stack_plan
from lb_anythings.domain.errors import ServiceDidNotStart

LABEL_STUDIO = "C:/tools/label-studio.exe"


@pytest.fixture(autouse=True)
def somewhere_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)  # no .env, and `data` lands here
    monkeypatch.setattr(shutil, "which", lambda name: LABEL_STUDIO)
    return tmp_path


def plan(settings: Settings | None = None, **options: object) -> StackPlan:
    return stack_plan(settings or Settings(), StackOptions(**options))  # type: ignore[arg-type]


def command_of(built: StackPlan, name: str) -> tuple[str, ...]:
    return next(service.command for service in built.services if service.name == name)


# --- which Services ---


def test_the_stack_is_label_studio_the_tracking_ui_and_the_backend_in_that_order() -> None:
    """Label Studio first because it is the slowest; the backend last, and stopped first."""
    assert [service.name for service in plan().services] == ["label-studio", "mlflow", "backend"]


def test_label_studio_can_be_left_out_for_one_that_is_already_running() -> None:
    assert "label-studio" not in [s.name for s in plan(label_studio=False).services]


def test_the_tracking_ui_is_left_out_when_nothing_is_being_tracked() -> None:
    assert "mlflow" not in [s.name for s in plan(Settings(tracking=False)).services]


def test_a_tracking_server_elsewhere_is_not_started_here() -> None:
    """LB_TRACKING_URI pointing at a server means someone else runs the UI."""
    settings = Settings(tracking_uri="http://mlflow.local:5000")

    assert "mlflow" not in [s.name for s in plan(settings).services]


# --- how each Service is started ---


def test_label_studio_is_started_on_the_port_it_is_expected_at_without_a_browser() -> None:
    settings = Settings(label_studio_url="http://localhost:8081/")

    command = command_of(plan(settings), "label-studio")

    assert command[0] == LABEL_STUDIO
    assert command[1] == "start"
    assert "--no-browser" in command  # `up` opens the project page itself, once it exists
    assert command[command.index("--port") + 1] == "8081"


def test_label_studio_is_expected_where_it_is_configured() -> None:
    settings = Settings(label_studio_url="http://localhost:8081")

    service = next(s for s in plan(settings).services if s.name == "label-studio")

    assert service.url == "http://localhost:8081"
    assert service.health_url == "http://localhost:8081/health"


def test_how_label_studio_is_started_can_be_replaced_wholesale() -> None:
    settings = Settings(label_studio_command="docker start -a my-label-studio")

    command = command_of(plan(settings), "label-studio")

    assert command == ("docker", "start", "-a", "my-label-studio")


def test_a_label_studio_that_cannot_be_found_says_both_ways_to_fix_it(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(shutil, "which", lambda name: None)

    with pytest.raises(ServiceDidNotStart, match="LB_LABEL_STUDIO_COMMAND"):
        plan()


def test_the_tracking_ui_serves_the_store_the_training_runs_write_to(tmp_path: Path) -> None:
    command = command_of(plan(), "mlflow")

    store = command[command.index("--backend-store-uri") + 1]
    artifacts = command[command.index("--default-artifact-root") + 1]
    assert store == f"sqlite:///{(tmp_path / 'data/mlflow/mlflow.db').as_posix()}"
    assert Path(artifacts) == tmp_path / "data/mlflow/artifacts"
    assert command[:4] == (sys.executable, "-m", "mlflow", "ui")


def test_the_tracking_ui_is_served_where_it_is_expected() -> None:
    settings = Settings(tracking_ui_url="http://127.0.0.1:5555")

    command = command_of(plan(settings), "mlflow")

    assert command[command.index("--port") + 1] == "5555"
    assert command[command.index("--host") + 1] == "127.0.0.1"


def test_the_backend_is_this_interpreter_serving_where_it_was_asked_to(tmp_path: Path) -> None:
    settings = Settings(host="0.0.0.0", port=9091)

    built = plan(settings)
    command = command_of(built, "backend")
    service = next(s for s in built.services if s.name == "backend")

    assert command[:4] == (sys.executable, "-m", "lb_anythings", "serve")
    assert command[command.index("--host") + 1] == "0.0.0.0"
    assert command[command.index("--port") + 1] == "9091"
    assert service.url == "http://127.0.0.1:9091", "0.0.0.0 is a bind address, not somewhere to ask"
    assert service.environment["LB_DATA_DIR"] == str(tmp_path / "data")


def test_the_backend_serves_where_the_flags_say(tmp_path: Path) -> None:
    command = command_of(plan(host="127.0.0.1", port=9099), "backend")

    assert command[command.index("--port") + 1] == "9099"


# --- setting up a Label Studio that has no user yet ---


def test_no_credentials_are_forced_on_a_label_studio_that_already_has_a_user() -> None:
    service = next(s for s in plan().services if s.name == "label-studio")

    assert "LABEL_STUDIO_USERNAME" not in service.environment


def test_a_first_user_is_created_from_the_environment_not_the_command_line() -> None:
    """A password in the command line is visible to everything with a process list."""
    settings = Settings(
        label_studio_username="me@example.com",
        label_studio_password=SecretStr("hunter2"),
        label_studio_api_key=SecretStr("a-legacy-token"),
    )

    service = next(s for s in plan(settings).services if s.name == "label-studio")

    assert "hunter2" not in " ".join(service.command)
    assert service.environment["LABEL_STUDIO_USERNAME"] == "me@example.com"
    assert service.environment["LABEL_STUDIO_PASSWORD"] == "hunter2"
    assert service.environment["LABEL_STUDIO_USER_TOKEN"] == "a-legacy-token"
    assert service.environment["LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN"] == "true"


def test_a_personal_access_token_is_not_handed_over_as_a_legacy_one() -> None:
    """Label Studio only sets a legacy token this way, and a JWT is not one."""
    settings = Settings(
        label_studio_username="me@example.com",
        label_studio_password=SecretStr("hunter2"),
        label_studio_api_key=SecretStr("header.payload.signature"),
    )

    service = next(s for s in plan(settings).services if s.name == "label-studio")

    assert "LABEL_STUDIO_USER_TOKEN" not in service.environment
    assert "LABEL_STUDIO_ENABLE_LEGACY_API_TOKEN" not in service.environment


# --- which project ---


def test_without_a_token_nothing_is_wired() -> None:
    """There is no way to ask Label Studio anything; the Services still come up."""
    assert plan().project is None


def test_the_project_is_the_configured_title_pointed_at_this_backend() -> None:
    settings = Settings(label_studio_api_key=SecretStr("secret"), project_title="Pipes", port=9091)

    project = plan(settings).project

    assert project is not None
    assert (project.title, project.model_url) == ("Pipes", "http://localhost:9091")


def test_a_label_studio_somewhere_else_is_told_where_this_backend_really_is() -> None:
    settings = Settings(
        label_studio_api_key=SecretStr("secret"), backend_url="http://host.docker.internal:9090"
    )

    project = plan(settings).project

    assert project is not None and project.model_url == "http://host.docker.internal:9090"


def test_a_new_project_offers_the_labels_the_training_set_already_uses(tmp_path: Path) -> None:
    """A rewired project should propose the classes the Examples on disk are labelled with."""
    examples = tmp_path / "data/examples"
    examples.mkdir(parents=True)
    (examples / "classes.txt").write_text("pipe\nflange\n", encoding="utf-8")

    project = plan(Settings(label_studio_api_key=SecretStr("secret"))).project

    assert project is not None and project.labels == ("pipe", "flange")


def test_the_labels_can_be_given_on_the_command_line() -> None:
    settings = Settings(label_studio_api_key=SecretStr("secret"))

    project = plan(settings, labels=("car", "truck")).project

    assert project is not None and project.labels == ("car", "truck")


def test_a_project_with_nothing_to_go_on_gets_one_generic_label() -> None:
    project = plan(Settings(label_studio_api_key=SecretStr("secret"))).project

    assert project is not None and project.labels == ("object",)


def test_the_configured_project_id_wins_over_the_title() -> None:
    settings = Settings(label_studio_api_key=SecretStr("secret"), project=42)

    project = plan(settings).project

    assert project is not None and project.project_id == 42


def test_the_flag_wins_over_the_configured_project_id() -> None:
    settings = Settings(label_studio_api_key=SecretStr("secret"), project=42)

    project = plan(settings, project_id=7).project

    assert project is not None and project.project_id == 7


def test_asking_for_a_new_project_says_so() -> None:
    project = plan(Settings(label_studio_api_key=SecretStr("secret")), always_create=True).project

    assert project is not None and project.always_create is True


def test_wiring_can_be_skipped_entirely() -> None:
    assert plan(Settings(label_studio_api_key=SecretStr("secret")), wire=False).project is None


def test_the_deadline_is_the_configured_one() -> None:
    assert plan(Settings(stack_timeout=42.0)).ready_timeout == 42.0


def test_stopping_the_backend_does_not_stop_a_training_run_it_launched() -> None:
    """A Training Run is detached on purpose (ADR 0002) and outlives the Stack."""
    services = {service.name: service for service in plan().services}

    assert services["backend"].stop_descendants is False
    assert services["label-studio"].stop_descendants is True
    assert services["mlflow"].stop_descendants is True
