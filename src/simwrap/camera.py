"""Single-thread-owned MuJoCo camera rendering."""

import threading
import time

import mujoco
import numpy as np

from .sim import Sim


class CameraRenderer:
    """Render named cameras on one worker thread and retain the latest frames."""

    def __init__(self, sim: Sim, cameras: tuple[str, ...] = ("front", "top", "wrist"),
                 width: int = 480, height: int = 360, rate_hz: float = 10.0):
        """Configure feeds and start their dedicated renderer thread."""
        if width <= 0 or height <= 0 or not np.isfinite(rate_hz) or rate_hz <= 0:
            raise ValueError("Resolution and frame rate must be positive")
        if len(cameras) != len(set(cameras)):
            raise ValueError("Camera names must be unique")
        available = set(sim.get_camera_names())
        unknown = set(cameras) - available
        if unknown:
            raise ValueError(f"Unknown cameras: {', '.join(sorted(unknown))}")
        self.sim = sim
        self.cameras = cameras
        self.width = width
        self.height = height
        self.period = 1.0 / rate_hz
        self._lock = threading.Lock()
        self._lifecycle_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._images: dict[str, np.ndarray] = {}
        self._generation = 0
        self._error: Exception | None = None
        self._closed = False
        self._thread = threading.Thread(target=self._run, name="mujoco-cameras", daemon=True)
        sim.set_camera_renderer(self)
        self._thread.start()

    def _run(self) -> None:
        """Own the renderer for its complete lifetime and report worker errors."""
        renderer: mujoco.Renderer | None = None
        try:
            renderer = self.sim.create_renderer(self.width, self.height)
            deadline = time.perf_counter()
            while not self._stop_event.is_set():
                frames = self.sim.render_cameras(renderer, self.cameras)
                with self._lock:
                    self._images = frames
                    self._generation += 1
                deadline += self.period
                now = time.perf_counter()
                if now - deadline > self.period:
                    deadline = now
                self._stop_event.wait(max(0.0, deadline - time.perf_counter()))
        except Exception as exc:
            self._error = exc
            self._stop_event.set()
        finally:
            if renderer is not None:
                renderer.close()

    def get_images(self) -> dict[str, np.ndarray]:
        """Return owned copies of the most recent named camera frames."""
        with self._lock:
            return {name: image.copy() for name, image in self._images.items()}

    def get_image(self, name: str) -> tuple[int, np.ndarray] | None:
        """Return one owned frame and its generation, or ``None`` before first render."""
        with self._lock:
            image = self._images.get(name)
            return None if image is None else (self._generation, image.copy())

    def get_error(self) -> str | None:
        """Return a renderer worker error if initialization or rendering failed."""
        return None if self._error is None else repr(self._error)

    def close(self) -> None:
        """Stop the worker and release its renderer; safe to call more than once."""
        with self._lifecycle_lock:
            if self._closed:
                return
            self._closed = True
            self._stop_event.set()
            if self._thread is not threading.current_thread():
                self._thread.join()

