"""Bringing the Stack up: start what is not running, wait for it, wire the project to it.

Every Service and every Label Studio call is a fake here, so the whole ordering -- adopt,
launch, wait, wire, and stop again -- is checked without a process or a socket.
"""

import pytest

from lb_anythings.application.ports import ConnectedModel, ServiceCommand
from lb_anythings.application.use_cases.bring_up_stack import (
    BringUpStack,
    ProjectPlan,
    StackPlan,
    shut_down,
)
from lb_anythings.domain.annotation_target import parse_label_config
from lb_anythings.domain.errors import ProjectWiringFailed, ServiceDidNotStart
from tests.fakes import FakeClock, FakeHealthProbe, FakeProjectAdmin, FakeServiceLauncher

LABEL_STUDIO = ServiceCommand(
    name="label-studio", command=("label-studio", "start"), url="http://localhost:8080"
)
MLFLOW = ServiceCommand(name="mlflow", command=("mlflow", "ui"), url="http://127.0.0.1:5000")
BACKEND = ServiceCommand(name="backend", command=("lb-anythings", "serve"), url="http://x:9090")

MODEL_URL = "http://localhost:9090"


def a_project(**overrides: object) -> ProjectPlan:
    defaults: dict[str, object] = {
        "title": "LB-anythings",
        "model_url": MODEL_URL,
        "labels": ("pipe",),
    }
    return ProjectPlan(**{**defaults, **overrides})  # type: ignore[arg-type]


class Stand:
    """A launcher, a probe, an admin and a clock, all writing to one journal."""

    def __init__(self) -> None:
        self.journal: list[str] = []
        self.probe = FakeHealthProbe()
        self.launcher = FakeServiceLauncher(self.probe, self.journal)
        self.admin = FakeProjectAdmin(self.journal)
        self.clock = FakeClock()

    def bring_up(self) -> BringUpStack:
        return BringUpStack(
            self.launcher, self.probe, self.admin, sleep=self.clock.sleep, now=self.clock.now
        )


# --- starting, adopting and waiting ---


def test_a_service_that_is_not_running_is_launched() -> None:
    stand = Stand()

    stack = stand.bring_up()(StackPlan(services=(LABEL_STUDIO,)))

    assert [s.name for s in stand.launcher.launched] == ["label-studio"]
    assert stack.services[0].adopted is False


def test_a_service_already_answering_is_adopted_not_launched() -> None:
    """Running `up` beside a Label Studio someone else started must not start a second one."""
    stand = Stand()
    stand.probe.answering(LABEL_STUDIO.health_url)

    stack = stand.bring_up()(StackPlan(services=(LABEL_STUDIO,)))

    assert stand.launcher.launched == []
    assert stack.services[0].adopted is True


def test_a_slow_service_is_waited_for() -> None:
    stand = Stand()
    stand.launcher.answers_after["label-studio"] = 6  # migrations, on a first start

    stack = stand.bring_up()(StackPlan(services=(LABEL_STUDIO,), poll_seconds=0.5))

    assert stack.services[0].name == "label-studio"
    assert stand.clock.slept == [0.5] * 6


def test_a_service_that_never_answers_gives_up_at_the_deadline() -> None:
    stand = Stand()
    stand.launcher.never_answers.add("label-studio")

    with pytest.raises(ServiceDidNotStart, match="label-studio"):
        stand.bring_up()(StackPlan(services=(LABEL_STUDIO,), ready_timeout=5.0, poll_seconds=1.0))

    assert stand.clock.now() <= 6.0, "waited past the deadline"


def test_a_service_that_dies_is_not_waited_out() -> None:
    """It is not coming back, and its log is where the reason is."""
    stand = Stand()
    stand.launcher.dies.add("label-studio")

    with pytest.raises(ServiceDidNotStart, match="stopped"):
        stand.bring_up()(
            StackPlan(services=(LABEL_STUDIO,), ready_timeout=600.0, poll_seconds=1.0)
        )

    assert stand.clock.now() == 0.0, "a dead Service should not cost a single poll"


def test_a_failure_stops_what_was_already_started() -> None:
    """Half a Stack left running is worse than none: the next `up` would adopt the wreckage."""
    stand = Stand()
    stand.launcher.never_answers.add("mlflow")

    with pytest.raises(ServiceDidNotStart):
        stand.bring_up()(
            StackPlan(services=(LABEL_STUDIO, MLFLOW), ready_timeout=2.0, poll_seconds=1.0)
        )

    assert "stop label-studio" in stand.journal
    assert "stop mlflow" in stand.journal


def test_an_adopted_service_is_never_stopped() -> None:
    stand = Stand()
    stand.probe.answering(LABEL_STUDIO.health_url)
    stand.launcher.never_answers.add("mlflow")

    with pytest.raises(ServiceDidNotStart):
        stand.bring_up()(
            StackPlan(services=(LABEL_STUDIO, MLFLOW), ready_timeout=2.0, poll_seconds=1.0)
        )

    assert "stop label-studio" not in stand.journal


def test_shutting_down_stops_what_was_started_in_reverse_order() -> None:
    """The backend goes first: Label Studio outliving it briefly beats the other way round."""
    stand = Stand()
    stack = stand.bring_up()(StackPlan(services=(LABEL_STUDIO, MLFLOW, BACKEND)))
    stand.journal.clear()

    shut_down(stack)

    assert stand.journal == ["stop backend", "stop mlflow", "stop label-studio"]


def test_shutting_down_leaves_adopted_services_alone() -> None:
    stand = Stand()
    stand.probe.answering(LABEL_STUDIO.health_url)
    stack = stand.bring_up()(StackPlan(services=(LABEL_STUDIO, BACKEND)))
    stand.journal.clear()

    shut_down(stack)

    assert stand.journal == ["stop backend"]


# --- wiring the project ---


def test_nothing_is_wired_until_every_service_answers() -> None:
    """Label Studio health-checks the backend before it accepts it as a model."""
    stand = Stand()
    stand.launcher.answers_after["backend"] = 2

    stand.bring_up()(StackPlan(services=(LABEL_STUDIO, BACKEND), project=a_project()))

    assert stand.journal.index("launch backend") < stand.journal.index("create LB-anythings")


def test_with_no_project_plan_label_studio_is_never_asked_anything() -> None:
    stand = Stand()

    stack = stand.bring_up()(StackPlan(services=(LABEL_STUDIO,)))

    assert stack.project is None
    assert stand.admin.training_on_submit == []


def test_a_project_with_the_configured_title_is_adopted() -> None:
    stand = Stand()
    stand.admin.add(7, "LB-anythings")

    stack = stand.bring_up()(StackPlan(services=(BACKEND,), project=a_project()))

    assert stack.project is not None
    assert (stack.project.id, stack.project.created) == (7, False)
    assert "create LB-anythings" not in stand.journal


def test_a_project_is_created_when_there_is_none_with_that_title() -> None:
    stand = Stand()

    stack = stand.bring_up()(StackPlan(services=(BACKEND,), project=a_project(title="Pipes")))

    assert stack.project is not None
    assert (stack.project.title, stack.project.created) == ("Pipes", True)


def test_a_created_project_gets_a_config_the_backend_can_read() -> None:
    stand = Stand()

    stack = stand.bring_up()(
        StackPlan(services=(BACKEND,), project=a_project(labels=("car", "truck")))
    )

    assert stack.project is not None
    target = parse_label_config(stand.admin.configs[stack.project.id])
    assert target.labels == ("car", "truck")


def test_new_creates_a_project_even_when_the_title_is_taken() -> None:
    stand = Stand()
    stand.admin.add(7, "LB-anythings")

    stack = stand.bring_up()(
        StackPlan(services=(BACKEND,), project=a_project(always_create=True))
    )

    assert stack.project is not None and stack.project.id != 7
    assert stack.project.created is True


def test_a_project_named_by_id_is_used_as_it_is() -> None:
    stand = Stand()
    stand.admin.add(42, "Something Else Entirely")

    stack = stand.bring_up()(StackPlan(services=(BACKEND,), project=a_project(project_id=42)))

    assert stack.project is not None
    assert (stack.project.id, stack.project.title) == (42, "Something Else Entirely")


def test_a_project_id_that_does_not_exist_is_a_clear_failure() -> None:
    stand = Stand()

    with pytest.raises(ProjectWiringFailed, match="no Label Studio project 42"):
        stand.bring_up()(StackPlan(services=(BACKEND,), project=a_project(project_id=42)))


def test_a_wiring_failure_stops_the_services_it_started() -> None:
    stand = Stand()

    with pytest.raises(ProjectWiringFailed):
        stand.bring_up()(StackPlan(services=(BACKEND,), project=a_project(project_id=42)))

    assert stand.journal[-1] == "stop backend"


def test_the_wired_project_has_training_on_submit_switched_on() -> None:
    """Without it Label Studio sends nothing, and the whole loop is silently dead."""
    stand = Stand()
    stand.admin.add(7, "LB-anythings")

    stand.bring_up()(StackPlan(services=(BACKEND,), project=a_project()))

    assert stand.admin.training_on_submit == [7]


def test_the_backend_is_connected_as_the_projects_model() -> None:
    stand = Stand()
    stand.admin.add(7, "LB-anythings")

    stack = stand.bring_up()(StackPlan(services=(BACKEND,), project=a_project()))

    assert stand.admin.models[7] == [ConnectedModel(MODEL_URL, interactive=True)]
    assert stack.project is not None and stack.project.model_connected is True


def test_a_model_connected_now_is_asked_for_predictions_while_labelling() -> None:
    """Connecting it any other way leaves that off, and no box ever appears."""
    stand = Stand()
    stand.admin.add(7, "LB-anythings")

    stack = stand.bring_up()(StackPlan(services=(BACKEND,), project=a_project()))

    assert stack.project is not None
    assert stack.project.predictions_while_labelling is True


def test_a_model_someone_connected_by_hand_may_never_be_asked_anything() -> None:
    """Interactive preannotations off is reported, not overruled: it is a choice."""
    stand = Stand()
    stand.admin.add(7, "LB-anythings", models=[ConnectedModel(MODEL_URL, False)])

    stack = stand.bring_up()(StackPlan(services=(BACKEND,), project=a_project()))

    assert stack.project is not None
    assert stack.project.predictions_while_labelling is False
    connections = [entry for entry in stand.journal if entry.startswith("connect")]
    assert connections == []  # the model is left exactly as the Operator set it up


def test_a_model_already_connected_is_not_connected_twice() -> None:
    stand = Stand()
    connected = ConnectedModel(MODEL_URL, interactive=True)
    stand.admin.add(7, "LB-anythings", models=[connected])

    stack = stand.bring_up()(StackPlan(services=(BACKEND,), project=a_project()))

    assert stand.admin.models[7] == [connected]
    assert stack.project is not None and stack.project.model_connected is False


def test_the_same_model_url_written_differently_still_counts_as_connected() -> None:
    stand = Stand()
    trailing = ConnectedModel(f"{MODEL_URL}/", interactive=True)
    stand.admin.add(7, "LB-anythings", models=[trailing])

    stand.bring_up()(StackPlan(services=(BACKEND,), project=a_project()))

    assert stand.admin.models[7] == [trailing]


def test_another_model_on_the_project_is_left_alone() -> None:
    """Someone else's model backend is not ours to remove."""
    stand = Stand()
    theirs = ConnectedModel("http://localhost:7070", interactive=True)
    stand.admin.add(7, "LB-anythings", models=[theirs])

    stand.bring_up()(StackPlan(services=(BACKEND,), project=a_project()))

    assert stand.admin.models[7] == [theirs, ConnectedModel(MODEL_URL, True)]
