"""A Training Run as a detached subprocess with an on-disk run record (ADR 0002).

The record (`<runs>/<run name>.json`) holds the run id, the run name, the pid, the process
creation time and the start time. Status is derived from it: running while that process is
alive, succeeded once a Checkpoint newer than the start exists, failed otherwise. A machine
reboot leaves a dead pid and no new Checkpoint, which reads as failed and unblocks the next run.
"""

import json
import logging
import os
import subprocess
import sys
import threading
import time
import uuid
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import psutil

from lb_anythings.domain.checkpoint import Checkpoint
from lb_anythings.domain.errors import TrainingAlreadyActive
from lb_anythings.domain.training_run import RunStatus, TrainingRun, derive_status

logger = logging.getLogger(__name__)

CREATE_TIME_TOLERANCE_SECONDS = 5.0  # guards against pid reuse after a reboot


@dataclass(frozen=True)
class RunRecord:
    id: str
    run_name: str
    pid: int
    created_at: float  # the process's own creation time, to recognise a reused pid
    started_at: float


def detached_process_flags() -> dict[str, Any]:
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP | subprocess.DETACHED_PROCESS}
    return {"start_new_session": True}


class SubprocessTrainer:
    def __init__(
        self, *, command: Sequence[str], runs_dir: Path, run_name: str, checkpoint: Path
    ) -> None:
        self._command = list(command)
        self._run_name = run_name
        self._record_file = runs_dir / f"{run_name}.json"
        self._log = runs_dir / f"{run_name}.log"
        self._checkpoint = checkpoint
        self._process: subprocess.Popen[bytes] | None = None
        self._start_lock = threading.Lock()

    def start(self, tracked_as: str | None = None) -> TrainingRun:
        with self._start_lock:  # two callers crossing a threshold together launch one run
            return self._start(tracked_as)

    def _start(self, tracked_as: str | None) -> TrainingRun:
        if self.active() is not None:
            raise TrainingAlreadyActive("a Training Run is already active; wait for it to finish")
        self._record_file.parent.mkdir(parents=True, exist_ok=True)
        started_at = time.time()
        # The recorded run travels as an argument, not an environment variable: it is visible
        # in the process list, and the child decides what it means.
        command = self._command + (["--tracked-as", tracked_as] if tracked_as else [])
        with self._log.open("ab") as log:
            self._process = subprocess.Popen(
                command,
                stdin=subprocess.DEVNULL,
                stdout=log,
                stderr=subprocess.STDOUT,
                **detached_process_flags(),
            )
        try:
            created_at = psutil.Process(self._process.pid).create_time()
        except psutil.Error:
            logger.debug("could not read the creation time of pid %s", self._process.pid)
            created_at = started_at
        run = TrainingRun(uuid.uuid4().hex, started_at, RunStatus.RUNNING)
        self._write(RunRecord(run.id, self._run_name, self._process.pid, created_at, started_at))
        logger.info("started Training Run %s as pid %s", run.id, self._process.pid)
        return run

    def active(self) -> TrainingRun | None:
        record = self._read()
        if record is None:
            return None
        run = self.refresh(TrainingRun(record.id, record.started_at, RunStatus.RUNNING))
        return run if run.is_active else None

    def refresh(self, run: TrainingRun) -> TrainingRun:
        record = self._read()
        alive = record is not None and record.id == run.id and self._alive(record)
        checkpoint = self._current_checkpoint()
        status = derive_status(
            process_alive=alive, started_at=run.started_at, checkpoint=checkpoint
        )
        return TrainingRun(
            run.id, run.started_at, status, checkpoint if status is RunStatus.SUCCEEDED else None
        )

    def _alive(self, record: RunRecord) -> bool:
        if self._process is not None and self._process.pid == record.pid:
            return self._process.poll() is None  # our own child: reap it if it has exited
        try:
            process = psutil.Process(record.pid)
            if not process.is_running() or process.status() == psutil.STATUS_ZOMBIE:
                return False
            drift = abs(process.create_time() - record.created_at)
        except psutil.Error:
            return False
        return drift < CREATE_TIME_TOLERANCE_SECONDS

    def _current_checkpoint(self) -> Checkpoint | None:
        if not self._checkpoint.is_file():
            return None
        return Checkpoint(self._checkpoint, self._checkpoint.stat().st_mtime)

    def _read(self) -> RunRecord | None:
        if not self._record_file.is_file():
            return None
        try:
            raw = json.loads(self._record_file.read_text(encoding="utf-8"))
            return RunRecord(
                id=str(raw["id"]),
                run_name=str(raw.get("run_name", self._run_name)),
                pid=int(raw["pid"]),
                created_at=float(raw["created_at"]),
                started_at=float(raw["started_at"]),
            )
        except (OSError, ValueError, KeyError, TypeError):
            logger.warning("run record %s is unreadable; treating it as no run", self._record_file)
            return None

    def _write(self, record: RunRecord) -> None:
        partial = self._record_file.with_name(f"{self._record_file.name}.{os.getpid()}.part")
        partial.write_text(json.dumps(asdict(record)), encoding="utf-8")
        os.replace(partial, self._record_file)
