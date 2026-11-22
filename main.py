"""Run the SO-101 scene with the dummy sinusoidal controller."""

import time
import sys
from pathlib import Path

# Allow running directly from a source checkout without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from simwrap.controller import DummyController
from simwrap.sim import Sim
from simwrap.ui import SimulationUI


def main() -> None:
    """Run the SO-101 simulation and viser browser viewer."""
    sim = Sim()
    controller = DummyController(sim)
    ui = SimulationUI(sim)
    sim.start()
    controller.start()
    ui.start()
    print("Viser viewer: http://localhost:8080")
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
