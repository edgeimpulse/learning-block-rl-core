import gymnasium  # noqa: F401
import mujoco  # noqa: F401

from ei_rl import sim
from ei_rl.sim import native  # noqa: F401
from toy_sim import XML  # noqa: F401


class Bad(sim.Task):
    pass
