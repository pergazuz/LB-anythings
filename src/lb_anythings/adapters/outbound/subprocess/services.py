"""A Service of the Stack as a child process, with its output tee'd to a log and the console.

Two things this has to get right, both of them Windows things. A Service is started in its own
process group, so the Ctrl+C that ends `up` does not race the orderly shutdown that follows it
-- the parent decides when each Service stops, in what order. And stopping one stops the
processes it spawned: Label Studio and MLflow both fork workers, and a worker that outlives its
parent keeps the port and makes the next `up` adopt a Stack that is not there.

Not every Service, though. A Training Run is a child of the backend and is meant to outlive it
(ADR 0002): it writes a Checkpoint the next `up` will serve, and killing it halfway through
wastes the GPU minutes and produces nothing. `stop_descendants` is what says which is which.
"""

import logging
import os
import subprocess
import sys
import threading
from collections.abc import Callable
from pathlib import Path
from typing import IO, Any

import psutil

from lb_anythings.application.ports import ServiceCommand

logger = logging.getLogger(__name__)

STOP_TIMEOUT_SECONDS = 10.0  # then it is killed


def own_process_group() -> dict[str, Any]:
    if sys.platform == "win32":
        return {"creationflags": subprocess.CREATE_NEW_PROCESS_GROUP}
    return {"start_new_session": True}


class LocalService:
    """One running Service. Stopping it stops its whole tree and closes its log."""

    def __init__(
        self,
        name: str,
        process: subprocess.Popen[bytes],
        log: IO[bytes],
        stop_descendants: bool = True,
    ) -> None:
        self.name = name
        self._process = process
        self._log = log
        self._stop_descendants = stop_descendants

    def running(self) -> bool:
        return self._process.poll() is None

    def stop(self) -> None:
        if self.running():
            _stop(self._process.pid, self.name, self._stop_descendants)
        try:
            self._process.wait(timeout=STOP_TIMEOUT_SECONDS)
        except subprocess.TimeoutExpired:  # pragma: no cover - the kill above should have won
            logger.warning("%s did not exit after being killed", self.name)
        if not self._log.closed:
            self._log.close()


class LocalServiceLauncher:
    """Starts Services as child processes, one append-only log file each."""

    def __init__(self, log_dir: Path, echo: Callable[[str], None] | None = None) -> None:
        self._log_dir = log_dir
        self._echo = echo

    def launch(self, service: ServiceCommand) -> LocalService:
        self._log_dir.mkdir(parents=True, exist_ok=True)
        log = (self._log_dir / f"{service.name}.log").open("ab")
        try:
            process = subprocess.Popen(
                list(service.command),
                stdin=subprocess.DEVNULL,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                env={**os.environ, **service.environment} if service.environment else None,
                **own_process_group(),
            )
        except OSError:
            log.close()
            raise
        logger.info("started %s as pid %s", service.name, process.pid)
        threading.Thread(
            target=self._pump, args=(service.name, process, log), daemon=True
        ).start()
        return LocalService(service.name, process, log, service.stop_descendants)

    def _pump(self, name: str, process: subprocess.Popen[bytes], log: IO[bytes]) -> None:
        """Copy the Service's output to its log and, if asked, to the console.

        A daemon thread per Service: reading the pipe is also what keeps a chatty Service from
        blocking on a full one.
        """
        assert process.stdout is not None
        try:
            for line in process.stdout:
                if not log.closed:
                    log.write(line)
                    log.flush()
                if self._echo is not None:
                    self._echo(f"[{name}] {line.decode('utf-8', 'replace').rstrip()}")
        except (OSError, ValueError):  # the pipe or the log closed under us, on the way down
            pass


def _stop(pid: int, name: str, descendants: bool) -> None:
    """Terminate a Service -- and, unless told otherwise, everything it spawned."""
    try:
        parent = psutil.Process(pid)
        # Both calls, under one guard: a Service that exits between them -- which is exactly
        # what a Service that failed to start does -- makes either of them raise.
        family = (parent.children(recursive=True) if descendants else []) + [parent]
    except psutil.Error:
        return
    for process in family:
        try:
            process.terminate()
        except psutil.Error:
            pass
    _, alive = psutil.wait_procs(family, timeout=STOP_TIMEOUT_SECONDS)
    for process in alive:
        logger.warning("%s (pid %s) ignored terminate; killing it", name, process.pid)
        try:
            process.kill()
        except psutil.Error:
            pass
