"""Starting and stopping a Service for real: actual child processes, actual log files.

Everything here spawns `sys.executable`, because the one thing worth testing about a process
launcher is what a real process does -- especially on Windows, where stopping one does not
stop what it spawned unless someone goes looking.
"""

import sys
import time
from collections.abc import Callable
from pathlib import Path

import psutil

from lb_anythings.adapters.outbound.subprocess.services import LocalServiceLauncher
from lb_anythings.application.ports import ServiceCommand

PATIENCE = 15.0  # a Python interpreter is slow to start on Windows, and CI is slower


def python(*script: str) -> tuple[str, ...]:
    return (sys.executable, "-c", "\n".join(script))


def service(name: str, *script: str, **fields: object) -> ServiceCommand:
    return ServiceCommand(name=name, command=python(*script), url="http://x", **fields)  # type: ignore[arg-type]


def until(condition: Callable[[], bool], patience: float = PATIENCE) -> bool:
    deadline = time.monotonic() + patience
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


SLEEPS = ("import time", "time.sleep(120)")


def test_a_launched_service_is_running(tmp_path: Path) -> None:
    process = LocalServiceLauncher(tmp_path).launch(service("sleeper", *SLEEPS))

    try:
        assert process.running()
    finally:
        process.stop()


def test_stopping_it_stops_it(tmp_path: Path) -> None:
    process = LocalServiceLauncher(tmp_path).launch(service("sleeper", *SLEEPS))

    process.stop()

    assert not process.running()


def test_stopping_one_that_has_already_stopped_is_not_an_error(tmp_path: Path) -> None:
    process = LocalServiceLauncher(tmp_path).launch(service("quick", "pass"))
    assert until(lambda: not process.running())

    process.stop()
    process.stop()


def test_a_service_that_exits_stops_running(tmp_path: Path) -> None:
    """This is what tells `up` that a Service died rather than being slow to answer."""
    process = LocalServiceLauncher(tmp_path).launch(service("quick", "raise SystemExit(3)"))

    assert until(lambda: not process.running())


def test_what_a_service_prints_lands_in_its_own_log(tmp_path: Path) -> None:
    launcher = LocalServiceLauncher(tmp_path)
    process = launcher.launch(service("noisy", "print('hello from the service')"))

    log = tmp_path / "noisy.log"
    assert until(lambda: log.is_file() and "hello from the service" in log.read_text("utf-8"))
    process.stop()


def test_what_it_prints_on_stderr_lands_there_too(tmp_path: Path) -> None:
    """Tracebacks are the interesting half, and they do not come out on stdout."""
    launcher = LocalServiceLauncher(tmp_path)
    process = launcher.launch(service("failing", "import sys", "print('boom', file=sys.stderr)"))

    log = tmp_path / "failing.log"
    assert until(lambda: log.is_file() and "boom" in log.read_text("utf-8"))
    process.stop()


def test_its_output_is_echoed_with_its_name(tmp_path: Path) -> None:
    echoed: list[str] = []
    launcher = LocalServiceLauncher(tmp_path, echo=echoed.append)
    process = launcher.launch(service("noisy", "print('up on 8080')"))

    assert until(lambda: any("up on 8080" in line for line in echoed))
    assert all(line.startswith("[noisy] ") for line in echoed)
    process.stop()


def test_the_service_gets_the_environment_it_was_given(tmp_path: Path) -> None:
    launcher = LocalServiceLauncher(tmp_path)
    process = launcher.launch(
        service(
            "env",
            "import os",
            "print(os.environ['LB_TEST_MARKER'], os.environ.get('PATH') is not None)",
            environment={"LB_TEST_MARKER": "set-by-the-launcher"},
        )
    )

    log = tmp_path / "env.log"
    assert until(lambda: log.is_file() and "set-by-the-launcher True" in log.read_text("utf-8"))
    process.stop()


def test_a_second_run_does_not_lose_the_first_ones_log(tmp_path: Path) -> None:
    launcher = LocalServiceLauncher(tmp_path)
    first = launcher.launch(service("noisy", "print('first run')"))
    log = tmp_path / "noisy.log"
    assert until(lambda: log.is_file() and "first run" in log.read_text("utf-8"))
    first.stop()

    second = launcher.launch(service("noisy", "print('second run')"))

    assert until(lambda: "second run" in log.read_text("utf-8"))
    assert "first run" in log.read_text("utf-8")
    second.stop()


def test_stopping_a_service_stops_what_it_spawned(tmp_path: Path) -> None:
    """Label Studio and MLflow both spawn workers; leaving them holding the port is the bug."""
    marker = tmp_path / "grandchild.pid"
    launcher = LocalServiceLauncher(tmp_path)
    process = launcher.launch(
        service(
            "parent",
            "import subprocess, sys, time, pathlib",
            "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(120)'])",
            f"pathlib.Path(r'{marker}').write_text(str(child.pid))",
            "time.sleep(120)",
        )
    )
    assert until(lambda: marker.is_file() and marker.read_text().strip().isdigit())
    grandchild = int(marker.read_text().strip())
    assert psutil.pid_exists(grandchild)

    process.stop()

    assert until(lambda: not _alive(grandchild), patience=20.0), "the grandchild outlived the stop"


def _alive(pid: int) -> bool:
    try:
        process = psutil.Process(pid)
        return process.is_running() and process.status() != psutil.STATUS_ZOMBIE
    except psutil.Error:
        return False


def test_a_service_whose_children_outlive_it_keeps_them(tmp_path: Path) -> None:
    """The backend's child is a Training Run: it writes a Checkpoint, so it is not ours to kill."""
    marker = tmp_path / "grandchild.pid"
    process = LocalServiceLauncher(tmp_path).launch(
        ServiceCommand(
            name="parent",
            command=python(
                "import subprocess, sys, time, pathlib",
                "child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(30)'])",
                f"pathlib.Path(r'{marker}').write_text(str(child.pid))",
                "time.sleep(120)",
            ),
            url="http://x",
            stop_descendants=False,
        )
    )
    assert until(lambda: marker.is_file() and marker.read_text().strip().isdigit())
    grandchild = int(marker.read_text().strip())

    process.stop()

    assert until(lambda: not process.running())
    assert _alive(grandchild), "the Training Run was killed with the Service that spawned it"
    psutil.Process(grandchild).kill()  # it would otherwise sit there for half a minute
