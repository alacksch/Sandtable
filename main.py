"""Run the SO-101 scene with the dummy sinusoidal controller."""

import time
import sys
import argparse
from pathlib import Path

# Allow running directly from a source checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from simwrap.controller import DummyController
from simwrap.sim import Sim
from simwrap.ui import SimulationUI


def main() -> None:
    """Run the simulation with the requested browser interface."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--ui", choices=("web", "viser"), default="viser")
    parser.add_argument("--port", type=int, default=8080, help="HTTP port for the selected UI (default: 8080)")
    args = parser.parse_args()
    sim = Sim()
    sim.start()
    if args.ui == "web":
        import uvicorn
        from simwrap.server import create_app
        app = create_app(sim)
        print(f"Web UI: http://localhost:{args.port}")
        try:
            uvicorn.run(app, host="0.0.0.0", port=args.port)
        finally:
            app.state.cameras.close()
            sim.close()
        return
    controller = DummyController(sim)
    ui = SimulationUI(sim, port=args.port)
    controller.start()
    ui.start()
    print(f"Viser viewer: http://localhost:{args.port}")
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        ui.close()
        sim.close()


if __name__ == "__main__":
    main()
