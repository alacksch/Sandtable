"""Single-thread-owned MuJoCo camera rendering."""

import threading
import time

import mujoco
import numpy as np

from .sim import Sim


class CameraRenderer:
    """Render named cameras on one worker thread and retain latest frames."""

    def __init__(self, sim: Sim, cameras: tuple[str, ...] = ("front", "top"),
                 width: int = 480, height: int = 360, rate_hz: float = 10.0):
        """Configure camera feeds and start their dedicated renderer thread."""
        if width <= 0 or height <= 0 or rate_hz <= 0:
            raise ValueError("Resolution and frame rate must be positive")
        self.sim = sim
        self.cameras = cameras
        self.width = width
        self.height = height
        self.period = 1.0 / rate_hz
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._images: dict[str, np.ndarray] = {}
        self._thread = threading.Thread(target=self._run, name="mujoco-cameras", daemon=True)
        sim.set_camera_renderer(self)
        self._thread.start()

    def _run(self) -> None:
        renderer = self.sim.create_renderer(self.width, self.height)
        try:
            while not self._stop_event.is_set():
                deadline = time.perf_counter() + self.period
                frames = {name: self.sim.render_camera(renderer, name) for name in self.cameras}
                with self._lock:
                    self._images = frames
                self._stop_event.wait(max(0.0, deadline - time.perf_counter()))
        finally:
            renderer.close()

    def get_images(self) -> dict[str, np.ndarray]:
        """Return owned copies of the most recent camera frames."""
        with self._lock:
            return {name: image.copy() for name, image in self._images.items()}

    def close(self) -> None:
        """Stop rendering and release the renderer on its owning thread."""
        self._stop_event.set()
        self._thread.join()

