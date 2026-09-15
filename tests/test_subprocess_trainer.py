"""Contract of the subprocess Trainer, driven with stub commands in place of real training."""

import json
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pytest

from lb_anythings.adapters.outbound.subprocess.trainer import SubprocessTrainer
from lb_anythings.domain.errors import TrainingAlreadyActive
from lb_anythings.domain.training_run import RunStatus

WRITES_CHECKPOINT = (
    "import sys, time, pathlib; time.sleep(0.5); p = pathlib.Path(sys.argv[1]); "
    "p.parent.mkdir(parents=True, exist_ok=True); p.write_bytes(b'weights')"
)
EXITS_WITHOUT_CHECKPOINT = "import sys; sys.exit(1)"
KEEPS_RUNNING = "import time; time.sleep(3)"
PRINTS = "print('hello from training')"


def _trainer(tmp_path: Path, script: str) -> SubprocessTrainer:
    checkpoint = tmp_path / "runs/active/weights/best.pt"
    return SubprocessTrainer(
        command=[sys.executable, "-c", script, str(checkpoint)],
        runs_dir=tmp_path / "runs",
        run_name="active",
        checkpoint=checkpoint,
    )


def _wait_until(condition: Callable[[], bool], timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return True
        time.sleep(0.05)
    return False


def test_a_run_is_active_until_its_process_ends_then_succeeded_with_the_checkpoint(
    tmp_path: Path,
) -> None:
    trainer = _trainer(tmp_path, WRITES_CHECKPOINT)

    run = trainer.start()

    active = trainer.active()
    assert active is not None and active.id == run.id
    assert _wait_until(lambda: trainer.active() is None)
    finished = trainer.refresh(run)
    assert finished.status is RunStatus.SUCCEEDED
    assert finished.checkpoint is not None and finished.checkpoint.path.name == "best.pt"


def test_a_process_that_exits_without_a_checkpoint_failed(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, EXITS_WITHOUT_CHECKPOINT)

    run = trainer.start()

    assert _wait_until(lambda: trainer.active() is None)
    assert trainer.refresh(run).status is RunStatus.FAILED


def test_a_second_run_is_refused_while_one_is_active(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, KEEPS_RUNNING)
    trainer.start()

    with pytest.raises(TrainingAlreadyActive):
        trainer.start()


def test_a_record_left_by_a_dead_process_does_not_block_a_new_run(tmp_path: Path) -> None:
    finished = subprocess.Popen([sys.executable, "-c", "pass"])
    finished.wait()
    runs = tmp_path / "runs"
    runs.mkdir()
    # a pid that is dead (or, if the OS reused it, was created long after this record says)
    stale = {"id": "old", "pid": finished.pid, "created_at": 0.0, "started_at": 0.0}
    (runs / "active.json").write_text(json.dumps(stale))
    trainer = _trainer(tmp_path, WRITES_CHECKPOINT)

    assert trainer.active() is None
    assert trainer.start().status is RunStatus.RUNNING


def test_the_process_output_lands_in_the_run_log(tmp_path: Path) -> None:
    trainer = _trainer(tmp_path, PRINTS)

    trainer.start()

    assert _wait_until(lambda: trainer.active() is None)
    assert "hello from training" in (tmp_path / "runs/active.log").read_text()


def test_two_callers_crossing_the_threshold_together_launch_one_run(tmp_path: Path) -> None:
    from concurrent.futures import ThreadPoolExecutor

    trainer = _trainer(tmp_path, KEEPS_RUNNING)

    def attempt() -> str:
        try:
            trainer.start()
            return "started"
        except TrainingAlreadyActive:
            return "refused"

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = sorted(pool.map(lambda _: attempt(), range(2)))

    assert outcomes == ["refused", "started"]


def test_the_spawned_run_is_handed_the_paths_rather_than_working_them_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The child re-reads the settings, so it must not be left to resolve them itself."""
    written = tmp_path / "seen.txt"
    script = f"import os; open({str(written)!r}, 'w').write(os.environ['LB_DATA_DIR'])"
    monkeypatch.chdir(tmp_path)
    trainer = SubprocessTrainer(
        command=[sys.executable, "-c", script],
        runs_dir=tmp_path / "runs",
        run_name="active",
        checkpoint=tmp_path / "runs/active/weights/best.pt",
        environment={"LB_DATA_DIR": str(tmp_path / "anchored")},
    )

    trainer.start()

    assert _wait_until(written.exists)
    assert written.read_text() == str(tmp_path / "anchored")


def test_the_spawned_run_keeps_the_rest_of_the_environment(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    written = tmp_path / "seen.txt"
    script = f"import os; open({str(written)!r}, 'w').write(os.environ.get('LB_CONF', 'gone'))"
    monkeypatch.setenv("LB_CONF", "0.4")
    trainer = SubprocessTrainer(
        command=[sys.executable, "-c", script],
        runs_dir=tmp_path / "runs",
        run_name="active",
        checkpoint=tmp_path / "runs/active/weights/best.pt",
        environment={"LB_DATA_DIR": str(tmp_path)},
    )

    trainer.start()

    assert _wait_until(written.exists)
    assert written.read_text() == "0.4"


def test_the_baseline_for_the_next_retrain_is_what_the_last_run_launched_on(
    tmp_path: Path,
) -> None:
    trainer = _trainer(tmp_path, WRITES_CHECKPOINT)
    assert trainer.trained_at_size() is None  # nothing has trained yet

    run = trainer.start(training_set_size=334)

    assert _wait_until(lambda: trainer.refresh(run).status is RunStatus.SUCCEEDED)
    assert trainer.trained_at_size() == 334


def test_a_failed_run_leaves_no_baseline_so_the_next_annotation_can_try_again(
    tmp_path: Path,
) -> None:
    """Nothing was learned from those Examples; waiting out a whole step would strand them."""
    trainer = _trainer(tmp_path, EXITS_WITHOUT_CHECKPOINT)

    run = trainer.start(training_set_size=334)

    assert _wait_until(lambda: trainer.refresh(run).status is RunStatus.FAILED)
    assert trainer.trained_at_size() is None
