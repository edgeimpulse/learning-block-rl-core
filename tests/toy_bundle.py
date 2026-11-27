# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Write a toy task's bundle, for the simulator extension's parity test:

  python tests/toy_bundle.py <out dir> [rail|drop]    # writes <out dir>/sim

rail (toy_sim.env): a cart on a rail, the loop and the page; drop (toy_sim.drop): contacts, the solver's
fields and an actuator that writes the model every physics step.
"""

import sys
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).parent))

from ei_rl import bundle  # noqa: E402
from toy_sim import XML, drop  # noqa: E402
from toy_sim.env import Rail  # noqa: E402
from toy_sim.robot import Cart  # noqa: E402

xml, task, robot = {"rail": (XML, Rail, Cart), "drop": (drop.XML, drop.Drop, drop.Arm)}[
    sys.argv[2] if len(sys.argv) > 2 else "rail"]
m = mujoco.MjModel.from_xml_string(xml)
vis_bin, visual = bundle.visual_model(m)
bundle.write_bundle(sys.argv[1], bundle.task_spec(task, robot, m), bundle.model_bytes(m), vis_bin, visual,
                    bundle.python_files(task, robot))
