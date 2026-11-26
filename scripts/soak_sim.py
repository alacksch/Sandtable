"""Exercise simulation/object concurrency and report bounded timing metrics."""

import argparse
import json
import statistics
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Callable

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from simwrap.objects import ObjectManager
from simwrap.sim import Sim


def percentile(values: list[float], fraction: float) -> float:
    """Return a nearest-rank percentile, or zero for an empty sample."""
    if not values:
        return 0.0
    ordered = sorted(values)
    return ordered[min(len(ordered) - 1, int((len(ordered) - 1) * fraction))]


def run_soak(duration: float) -> dict[str, object]:
    """Stress physics stepping with concurrent object edits and pool reuse."""
    sim = Sim()
    objects = ObjectManager(sim)
    stop = threading.Event()
    metric_lock = threading.Lock()
    operation_seconds: deque[float] = deque(maxlen=50_000)
    operation_count = 0
    worker_errors: list[str] = []
    names = [
        objects.spawn("box", np.array((-0.3, -0.32, 0.785))),
        objects.spawn("cylinder", np.array((0.0, -0.32, 0.79))),
        objects.spawn("sphere", np.array((0.3, -0.32, 0.785))),
    ]
    sim.start()
    deadline = time.perf_counter() + duration

    def measure(operation: Callable[[int], None]) -> None:
        """Repeat an operation until the deadline and retain its latency."""
        nonlocal operation_count
        try:
            index = 0
            while not stop.is_set():
                started = time.perf_counter()
                operation(index)
                elapsed = time.perf_counter() - started
                with metric_lock:
                    operation_seconds.append(elapsed)
                    operation_count += 1
                index += 1
        except Exception as exc:  # surface worker failures in the final report
            with metric_lock:
                worker_errors.append(repr(exc))
            stop.set()

    def edit(index: int) -> None:
        offset = 0.01 if index % 2 else -0.01
        objects.move(names[0], np.array((-0.3 + offset, -0.32, 0.785)),
                     np.array((1.0, 0.0, 0.0, 0.0)))

    def reuse(_: int) -> None:
        name = objects.spawn("sphere", np.array((0.48, 0.3, 0.785)))
        objects.delete(name)

    workers = [
        threading.Thread(target=measure, args=(edit,), name="soak-object-edit"),
        threading.Thread(target=measure, args=(reuse,), name="soak-pool-reuse"),
    ]
    try:
        for worker in workers:
            worker.start()
        physics_samples: deque[float] = deque(maxlen=50_000)
        previous_wall = time.perf_counter()
        previous_sim = sim.get_simulation_time()
        while time.perf_counter() < deadline and not stop.is_set():
            time.sleep(0.05)
            wall = time.perf_counter()
            sim_time = sim.get_simulation_time()
            physics_samples.append((sim_time - previous_sim) / (wall - previous_wall))
            previous_wall, previous_sim = wall, sim_time
        stop.set()
        for worker in workers:
            worker.join(timeout=2.0)
        lingering = [worker.name for worker in workers if worker.is_alive()]
        snapshot = sim.get_state_snapshot()
        worker_error = sim.get_worker_error()
        with metric_lock:
            latency_ms = [value * 1000 for value in operation_seconds]
            errors = list(worker_errors)
        result = {
            "duration_seconds": duration,
            "object_operations": operation_count,
            "latency_samples_retained": len(latency_ms),
            "operation_latency_ms": {
                "p50": percentile(latency_ms, 0.50),
                "p95": percentile(latency_ms, 0.95),
                "max": max(latency_ms, default=0.0),
            },
            "physics_time_ratio": {
                "mean": statistics.fmean(physics_samples) if physics_samples else 0.0,
                "min": min(physics_samples, default=0.0),
                "max": max(physics_samples, default=0.0),
            },
            "finite_state": all(np.isfinite(value).all() for value in snapshot.values()),
            "physics_worker_error": worker_error,
            "object_worker_errors": errors,
            "lingering_object_workers": lingering,
        }
    finally:
        stop.set()
        for worker in workers:
            if worker.ident is not None:
                worker.join(timeout=2.0)
        sim.close()
    result["physics_worker_stopped"] = not any(
        thread.name == "mujoco-physics" and thread.is_alive() for thread in threading.enumerate()
    )
    return result


def main() -> None:
    """Parse options and run the requested bounded soak interval."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seconds", type=float, default=30.0, help="Soak duration in seconds (default: 30)")
    args = parser.parse_args()
    if not np.isfinite(args.seconds) or args.seconds <= 0 or args.seconds > 3600:
        parser.error("--seconds must be positive and no more than 3600")
    result = run_soak(args.seconds)
    print(json.dumps(result, indent=2))
    if (not result["finite_state"] or result["physics_worker_error"] or
            result["object_worker_errors"] or result["lingering_object_workers"] or
            not result["physics_worker_stopped"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
