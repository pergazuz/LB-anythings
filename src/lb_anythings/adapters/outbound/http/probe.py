"""Is anything answering there yet? The readiness question, asked over and over."""

import httpx

# Short: this runs in a poll loop, and a Service that is still starting up does not answer at
# all rather than answering slowly.
PROBE_TIMEOUT_SECONDS = 3.0


class HttpHealthProbe:
    def __init__(self, http: httpx.Client | None = None) -> None:
        self._http = http or httpx.Client(timeout=PROBE_TIMEOUT_SECONDS, follow_redirects=True)

    def answers(self, url: str) -> bool:
        """True when something answered. A 404 counts: the Service is there, the path was ours.

        A 5xx does not: that is a Service that has come up broken, and waiting out the
        deadline and saying so beats wiring a project into it.
        """
        try:
            return self._http.get(url).status_code < 500
        except httpx.HTTPError:
            return False
