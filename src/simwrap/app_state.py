"""Thread-safe UI state shared by the web and legacy interfaces."""

import threading


class AppState:
    """Store the current prompt and policy run flag."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._prompt = ""
        self._running = False
        self._policy_status = "offline"

    def set_prompt(self, prompt: str) -> None:
        """Update the prompt text."""
        with self._lock:
            self._prompt = prompt

    def set_running(self, running: bool) -> None:
        """Set whether a policy run is active."""
        with self._lock:
            self._running = running

    def snapshot(self) -> tuple[str, bool]:
        """Return a consistent state snapshot."""
        with self._lock:
            return self._prompt, self._running

    def set_policy(self, *, running: bool, status: str) -> None:
        """Update policy lifecycle state for API clients."""
        with self._lock:
            self._running = running
            self._policy_status = status

    def policy_status(self) -> str:
        """Return the current policy lifecycle label."""
        with self._lock:
            return self._policy_status
