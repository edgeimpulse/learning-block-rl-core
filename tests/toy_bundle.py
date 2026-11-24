# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Write the toy task's bundle (toy_sim), for the simulator extension's parity test:

  python tests/toy_bundle.py <out dir>    # writes <out dir>/sim
"""

import sys
from pathlib import Path

import mujoco

sys.path.insert(0, str(Path(__file__).parent))

from ei_rl import bundle  # noqa: E402
from toy_sim import XML  # noqa: E402
from toy_sim.env import Rail  # noqa: E402
from toy_sim.robot import Cart  # noqa: E402

m = mujoco.MjModel.from_xml_string(XML)
vis_bin, visual = bundle.visual_model(m)
bundle.write_bundle(sys.argv[1], bundle.task_spec(Rail, Cart, m), bundle.model_bytes(m), vis_bin, visual,
                    bundle.python_files(Rail, Cart))
