"""Contract of the thread runner: the job runs elsewhere; its failure never reaches the caller."""

import threading

from lb_anythings.adapters.outbound.background.runner import ThreadBackgroundRunner


def test_the_job_runs_on_another_thread() -> None:
    done = threading.Event()
    seen: list[str] = []

    def job() -> None:
        seen.append(threading.current_thread().name)
        done.set()

    ThreadBackgroundRunner().run(job)

    assert done.wait(timeout=5)
    assert seen and seen[0] != threading.current_thread().name


def test_a_failing_job_does_not_raise_into_the_caller() -> None:
    done = threading.Event()

    def job() -> None:
        done.set()
        raise RuntimeError("boom")

    ThreadBackgroundRunner().run(job)  # would raise here if the failure escaped the thread

    assert done.wait(timeout=5)
