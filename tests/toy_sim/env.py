import numpy as np

from ei_rl.sim import Task, command, done, metric, point

from .robot import Cart  # noqa: F401  (a local import, which the bundle follows)


class Rail(Task):
    id = "Toy-Rail-v0"
    label = "Toy rail"
    decimation = 4
    episode_s = 2.0
    commands = [command("target", 1, -0.8, 0.8, 0.5, "Target", axes=["x"])]
    states = ["left", "right"]

    def __init__(self, sim):
        super().__init__(sim)
        self.q = sim.model.jnt_qposadr[sim.id("joint", "slide")]
        self.tip = sim.id("site", "tip")

    def reset(self, sim, rng, state=None):
        state = rng.integers(len(self.states)) if state is None else state
        sim.data.qpos[self.q] = (-0.5, 0.5)[state] + rng.uniform(-0.05, 0.05)
        self.prev = None

    def after_step(self, sim):
        self.prev = float(sim.data.qpos[self.q])

    def observe(self, sim, name, rng):
        x = np.array([sim.data.qpos[self.q], sim.data.qvel[self.q], self.command_values["target"][0],
                      sim.data.site_xpos[self.tip, 0]])
        if sim.options["noise"]:
            x = x + rng.normal(0, 0.01, x.shape)
        return x.astype(np.float32)

    def done(self, sim):
        if abs(sim.data.qpos[self.q]) > 1.5:
            return done("lost", "Off the rail", "failure")
        return super().done(sim)

    def metrics(self, sim):
        return [metric("Position", sim.data.qpos[self.q], "m")]

    def markers(self, sim):
        return [point([self.command_values["target"][0], 0, 0.1])]
