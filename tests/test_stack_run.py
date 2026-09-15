"""Running the Stack: what an Operator is told, and what happens when they stop it."""

import shutil
import webbrowser
from pathlib import Path

import pytest

from lb_anythings.application.use_cases.bring_up_stack import (
    RunningService,
    Stack,
    StackPlan,
    WiredProject,
)
from lb_anythings.bootstrap.settings import Settings
from lb_anythings.bootstrap.stack import StackOptions, run_stack, summary, watch
from lb_anythings.domain.errors import ProjectWiringFailed, ServiceDidNotStart
from tests.fakes import FakeService


@pytest.fixture(autouse=True)
def somewhere_clean(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(webbrowser, "open", lambda url: True)
    monkeypatch.setattr(shutil, "which", lambda name: "label-studio")  # the plan is not the point
    return tmp_path


def a_stack(project: WiredProject | None = None, journal: list[str] | None = None) -> Stack:
    backend = FakeService("backend", journal if journal is not None else [])
    started = RunningService("backend", "http://127.0.0.1:9090", adopted=False, process=backend)
    adopted = RunningService("label-studio", "http://localhost:8080", adopted=True)
    return Stack(services=(adopted, started), project=project, started=(started,))


WIRED = WiredProject(id=7, title="Pipes", created=True, model_connected=True)


# --- the summary ---


def test_the_summary_says_where_each_service_is_and_how_it_got_there() -> None:
    lines = summary(Settings(), StackOptions(), a_stack())

    assert any("label-studio" in line and "already running" in line for line in lines)
    assert any("backend" in line and "started" in line for line in lines)


def test_the_summary_links_straight_to_the_project() -> None:
    lines = summary(Settings(), StackOptions(), a_stack(WIRED))

    assert any("http://localhost:8080/projects/7/data" in line for line in lines)
    assert any("Pipes" in line and "created" in line for line in lines)


def test_a_project_that_was_already_there_is_not_announced_as_new() -> None:
    existing = WiredProject(id=7, title="Pipes", created=False, model_connected=False)

    lines = summary(Settings(), StackOptions(), a_stack(existing))

    assert any("existing" in line and "already connected" in line for line in lines)


def test_with_no_token_the_summary_says_exactly_what_to_do() -> None:
    lines = "\n".join(summary(Settings(), StackOptions(), a_stack()))

    assert "LABEL_STUDIO_API_KEY" in lines
    assert "Access Token" in lines


def test_asking_for_no_wiring_is_not_reported_as_a_missing_token() -> None:
    lines = "\n".join(summary(Settings(), StackOptions(wire=False), a_stack()))

    assert "LABEL_STUDIO_API_KEY" not in lines
    assert "--no-wiring" in lines


def test_a_model_label_studio_never_asks_is_called_out() -> None:
    """Everything else can be right and no box will ever appear on a Task."""
    quiet = WiredProject(7, "Pipes", created=False, model_connected=False,
                         predictions_while_labelling=False)

    lines = "\n".join(summary(Settings(), StackOptions(), a_stack(quiet)))

    assert "interactive preannotations are off" in lines


def test_nothing_is_said_when_predictions_do_appear() -> None:
    lines = "\n".join(summary(Settings(), StackOptions(), a_stack(WIRED)))

    assert "interactive preannotations" not in lines


def test_the_summary_says_where_the_logs_are(tmp_path: Path) -> None:
    lines = "\n".join(summary(Settings(), StackOptions(), a_stack()))

    assert str(tmp_path / "data" / "logs") in lines


# --- running until stopped ---


def _run(**kwargs: object) -> tuple[int, list[str], list[str]]:
    journal: list[str] = []
    printed: list[str] = []
    stack = a_stack(WIRED, journal)
    code = run_stack(
        Settings(),
        kwargs.pop("options", StackOptions()),  # type: ignore[arg-type]
        bring_up=kwargs.pop("bring_up", lambda plan: stack),  # type: ignore[arg-type]
        wait=kwargs.pop("wait", lambda s: None),  # type: ignore[arg-type]
        out=printed.append,
    )
    return code, printed, journal


def test_running_the_stack_reports_it_and_then_stops_it() -> None:
    code, printed, journal = _run()

    assert code == 0
    assert any("Pipes" in line for line in printed)
    assert journal == ["stop backend"]


def test_ctrl_c_is_a_clean_stop() -> None:
    def interrupted(stack: Stack) -> str | None:
        raise KeyboardInterrupt

    code, _, journal = _run(wait=interrupted)

    assert code == 0
    assert journal == ["stop backend"]


def test_a_service_dying_on_its_own_brings_the_rest_down_and_fails() -> None:
    code, _, journal = _run(wait=lambda stack: "mlflow")

    assert code == 1
    assert journal == ["stop backend"]


def test_ctrl_c_while_it_is_still_coming_up_is_not_a_crash(
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Label Studio takes half a minute to start; changing your mind is not an error."""

    def interrupted(plan: StackPlan) -> Stack:
        raise KeyboardInterrupt  # BringUpStack has already stopped what it started

    code, printed, _ = _run(bring_up=interrupted)

    assert code == 130
    assert any("stopped before the Stack was up" in line for line in printed)
    assert capsys.readouterr().err == ""


def test_a_stack_that_will_not_come_up_fails_without_a_traceback(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def refuses(plan: StackPlan) -> Stack:
        raise ServiceDidNotStart("label-studio did not answer")

    code, printed, _ = _run(bring_up=refuses)

    assert code == 1
    assert "label-studio did not answer" in capsys.readouterr().err
    assert printed == []


def test_a_project_that_cannot_be_wired_fails_the_same_way(
    capsys: pytest.CaptureFixture[str],
) -> None:
    def refuses(plan: StackPlan) -> Stack:
        raise ProjectWiringFailed("no Label Studio project 42")

    code, _, _ = _run(bring_up=refuses)

    assert code == 1
    assert "no Label Studio project 42" in capsys.readouterr().err


def _watch_the_browser(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    opened: list[str] = []

    def open_url(url: str) -> bool:
        opened.append(url)
        return True

    monkeypatch.setattr(webbrowser, "open", open_url)
    return opened


def test_the_project_is_opened_in_a_browser(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = _watch_the_browser(monkeypatch)

    _run()

    assert opened == ["http://localhost:8080/projects/7/data"]


def test_nothing_is_opened_when_the_operator_says_not_to(monkeypatch: pytest.MonkeyPatch) -> None:
    opened = _watch_the_browser(monkeypatch)

    _run(options=StackOptions(open_browser=False))

    assert opened == []


def test_watching_returns_the_service_that_stopped() -> None:
    journal: list[str] = []
    dead = FakeService("mlflow", journal)
    dead.stop()
    stack = Stack(started=(RunningService("mlflow", "http://x", False, dead),))

    assert watch(stack, interval=0.0) == "mlflow"
