"""Run the SO-101 scene with the dummy sinusoidal controller."""

import time

from simwrap.controller import DummyController
from simwrap.sim import Sim


def main() -> None:
    """Run the headless simulation until interrupted."""
    sim = Sim()
    controller = DummyController(sim)
    sim.start()
    controller.start()
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        controller.stop()
        sim.stop()


if __name__ == "__main__":
    main()
