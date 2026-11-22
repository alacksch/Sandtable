"""Simple sinusoidal actuator controller for scene bring-up."""

import threading
import time

import numpy as np

from .sim import Sim


class DummyController:
    """Send slow bounded sinusoidal targets to a running simulation."""

    def __init__(self, sim: Sim, frequency: float = 0.08, amplitude_fraction: float = 0.2):
        """Create a controller with a modest fraction of each joint's range."""
        self.sim = sim
        self.frequency = frequency
        self.amplitude_fraction = amplitude_fraction
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        """Start sending targets in a background thread."""
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, name="dummy-controller", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        """Stop and join the controller thread."""
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join()
            self._thread = None

    def _run(self) -> None:
        ranges = self.sim.get_ctrl_ranges()
        actuator_count = self.sim.get_actuator_count()
        centers = ranges.mean(axis=1)
        amplitudes = (ranges[:, 1] - ranges[:, 0]) * self.amplitude_fraction / 2
        start = time.perf_counter()
        while not self._stop_event.is_set():
            t = time.perf_counter() - start
            phases = np.arange(actuator_count) * (np.pi / 3)
            self.sim.set_ctrl(centers + amplitudes * np.sin(2 * np.pi * self.frequency * t + phases))
            self._stop_event.wait(0.02)
