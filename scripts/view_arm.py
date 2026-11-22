"""Open the official SO-101 scene in the MuJoCo viewer."""

from pathlib import Path

import mujoco
import mujoco.viewer


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "assets" / "so101" / "scene.xml"


def main() -> None:
    """Load the scene and open its interactive viewer."""
    model = mujoco.MjModel.from_xml_path(str(MODEL_PATH))
    data = mujoco.MjData(model)
    mujoco.viewer.launch(model, data)


if __name__ == "__main__":
    main()
