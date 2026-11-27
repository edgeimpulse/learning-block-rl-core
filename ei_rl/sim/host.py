# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""Host: the loop that steps a Task and its Robots, in the order the ei_rl.sim docstring gives. The same code
runs in training (ei_rl.sim.native.Host, on mujoco) and in the extension (ei_rl.sim.browser, on MuJoCo WASM).
It needs from its Sim, besides the contract: _reset_data() (mj_resetData), _step() (mj_step), forward().
"""

from collections import deque

import numpy as np

# action_delay, obs_delay: whole policy steps the host holds the actions and the observations back, on top of any
# latency the robot models itself (a what-if in the extension; 0 in training)
DEFAULT_OPTIONS = dict(noise=True, randomize=True, obs_delay=0, action_delay=0)


class Host:
    """Host(sim, task_cls, robot_cls, options=None, seed=0).

    options: the extension's switches (DEFAULT_OPTIONS) and the settings' values (their defaults if not
    given), into sim.options; seed: the rng's.

      host.reset(state=None)  -> {name: observation}
      host.step(actions)      -> ({name: observation}, done or None); actions: {name: action}, or the
                                 policy robot's action alone (the others then get zeros)
    """

    def __init__(self, sim, task_cls, robot_cls, options=None, seed=0):
        settings = {s["name"]: s["default"] for s in list(task_cls.settings) + list(robot_cls.settings)}
        sim.options = {**DEFAULT_OPTIONS, **settings, **(options or {})}  # options win over the defaults
        self.sim = sim
        self.names = [a["name"] for a in task_cls.agents]
        self.robots = {n: robot_cls(sim, n) for n in self.names}  # one per agent, before the task: it gets them
        self.task = task_cls(sim, self.robots)
        self.rng = np.random.default_rng(seed)
        self.action_dim = robot_cls.action_dim
        self.obs = None
        self.steps = 0

    def reset(self, state=None):
        sim, rng = self.sim, self.rng
        sim._reset_data()
        drawn = rng if sim.options["randomize"] else None
        for r in self.robots.values():
            r.randomize(sim, drawn)
        self.task.randomize(sim, drawn)
        self.task.reset(sim, rng, state)
        for r in self.robots.values():
            r.reset(sim, rng)
        sim.forward()
        zeros = np.zeros(self.action_dim)
        self.pending = deque([{n: zeros for n in self.names}] * int(sim.options["action_delay"] or 0))
        self.steps = 0
        self.obs = self.observe()
        self.seen = deque([self.obs] * int(sim.options["obs_delay"] or 0))  # the observations still on their way
        return self.obs

    def observe(self):
        return {n: self.task.observe(self.sim, n, self.rng) for n in self.names}

    def step(self, actions):
        sim = self.sim
        if not isinstance(actions, dict):
            actions = {self.names[0]: actions}
        actions = {n: np.asarray(actions.get(n, np.zeros(self.action_dim)), np.float64) for n in self.names}
        self.pending.append(actions)
        applied = self.pending.popleft()
        for n, r in self.robots.items():
            r.command(sim, applied[n])
        for _ in range(sim.decimation):
            for r in self.robots.values():
                r.apply(sim)
            sim._step()
        if self.task.forward_after_step:
            sim.forward()
        self.steps += 1
        self.task.after_step(sim)
        self.seen.append(self.observe())
        self.obs = self.seen.popleft()
        return self.obs, self.task.done(sim)

    def metrics(self):
        out = list(self.task.metrics(self.sim))
        labels = {a["name"]: a["label"] for a in self.task.agents}
        for n, r in self.robots.items():
            for m in r.metrics(self.sim):  # each robot's, under its name when there are several
                out.append(dict(m, label=f"{labels[n]} {m['label'][:1].lower()}{m['label'][1:]}") if len(self.robots) > 1 else m)
        return out

    def markers(self):
        return list(self.task.markers(self.sim))
