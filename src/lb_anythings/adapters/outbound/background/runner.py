"""Runs jobs on a daemon thread so webhook responses return at once."""

import logging
import threading
from collections.abc import Callable

logger = logging.getLogger(__name__)


class ThreadBackgroundRunner:
    def run(self, job: Callable[[], object]) -> None:
        def guarded() -> None:
            try:
                job()
            except Exception:
                logger.exception("background job failed")

        threading.Thread(target=guarded, daemon=True).start()
