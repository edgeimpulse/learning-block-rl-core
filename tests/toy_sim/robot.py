import numpy as np

from ei_rl.sim import Robot, metric, setting


class Cart(Robot):
    """A motor behind a first-order lag: the actuator state a reset must clear."""
    action_dim = 1
    keys = {"ArrowRight": [0, 1.0], "ArrowLeft": [0, -1.0]}
    settings = [setting("lag_s", 0.0, 0.2, 0.05, "Motor lag", "s")]

    def __init__(self, sim, name):
        super().__init__(sim, name)
        self.act = sim.id("actuator", "push")
        self.mass0 = float(sim.model.body_mass[sim.id("body", "cart")])

    def randomize(self, sim, rng):
        k = 1.0 if rng is None else rng.uniform(0.8, 1.2)
        sim.model.body_mass[sim.id("body", "cart")] = self.mass0 * k

    def reset(self, sim, rng):
        self.u, self.target = 0.0, 0.0

    def command(self, sim, action):
        self.target = float(np.clip(action[0], -1, 1))

    def apply(self, sim):
        lag = sim.options["lag_s"]
        a = 1.0 if lag <= 0 else min(sim.physics_dt / lag, 1.0)
        self.u += a * (self.target - self.u)
        sim.data.ctrl[self.act] = self.u

    def metrics(self, sim):
        return [metric("Motor", self.u)]
