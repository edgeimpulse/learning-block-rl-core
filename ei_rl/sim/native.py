# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""The simulation contract on mujoco: NativeSim, and Host, the loop that steps a Robot and a Task (see
ei_rl.sim). A block's training env steps its task through a Host, so it trains in the loop the extension
runs; from_bundle loads a written bundle the way the extension does, for the block's checks.
"""

import importlib
import json
import sys
from collections import deque
from pathlib import Path

import mujoco
import numpy as np

from . import DATA_FIELDS, MODEL_FIELDS, MODEL_SIZES, OBJECT_KINDS, Sim

_OBJ = {k: getattr(mujoco.mjtObj, "mjOBJ_" + ("JOINT" if k == "joint" else k.upper())) for k in OBJECT_KINDS}

DEFAULT_OPTIONS = dict(noise=True, randomize=True, obs_delay=None, action_delay=0)


class _Fields:
    """The contract's fields of an MjModel or MjData, and only those: what a block reads here, the browser
    has too. mujoco's arrays are views of buffers that live as long as the model or data, so each is looked
    up once."""

    def __init__(self, src, fields, scalars=()):
        object.__setattr__(self, "_src", src)
        object.__setattr__(self, "_fields", frozenset(fields) | frozenset(scalars))

    def __getattr__(self, k):
        if k not in self._fields:
            raise AttributeError(f"{k!r} is not in the simulation contract (ei_rl.sim MODEL_FIELDS, DATA_FIELDS)")
        v = getattr(self._src, k)
        object.__setattr__(self, k, v)
        return v

    def __setattr__(self, k, v):
        raise AttributeError(f"assign into sim.{k}[...], not the field itself")


class _Data(_Fields):
    @property
    def time(self):
        return self._src.time


class NativeSim(Sim):
    def __init__(self, model, data, decimation, options=None):
        self._m, self._d = model, data
        self.model = _Fields(model, MODEL_FIELDS, MODEL_SIZES)
        self.data = _Data(data, DATA_FIELDS)
        self.physics_dt = float(model.opt.timestep)
        self.decimation = int(decimation)
        self.options = dict(DEFAULT_OPTIONS, **(options or {}))

    def name2id(self, kind, name):
        return mujoco.mj_name2id(self._m, _OBJ[kind], name)

    def id2name(self, kind, i):
        return mujoco.mj_id2name(self._m, _OBJ[kind], i)

    def ray(self, pnt, vec, geomgroup=None, flg_static=True, bodyexclude=-1):
        gid = np.full(1, -1, np.int32)
        gg = None if geomgroup is None else np.asarray(geomgroup, np.uint8)
        dist = mujoco.mj_ray(self._m, self._d, np.asarray(pnt, np.float64), np.asarray(vec, np.float64), gg,
                             int(bool(flg_static)), int(bodyexclude), gid)
        return float(dist), int(gid[0])

    def forward(self):
        mujoco.mj_forward(self._m, self._d)


def check_fields(model, data):
    """The contract's shapes are mujoco's: a mismatch would read garbage in the browser."""
    for src, fields in ((model, MODEL_FIELDS), (data, DATA_FIELDS)):
        for k, (rows, cols) in fields.items():
            n = getattr(model, rows)
            want = (n,) if cols is None else (n, getattr(model, cols) if isinstance(cols, str) else cols)
            got = np.shape(getattr(src, k))
            if got != want and not (n == 0 and np.size(getattr(src, k)) == 0):
                raise AssertionError(f"{k}: mujoco's shape {got}, the contract's {want}")


class Host:
    """Steps a Task and its Robots on a model, in the order the ei_rl.sim docstring gives.

    task_cls, robot_cls: the classes; options: the extension's switches and settings (DEFAULT_OPTIONS, plus
    each setting's default); seed: the rng's seed.

      host.reset(state=None)  -> {name: observation}
      host.step(actions)      -> ({name: observation}, done or None); actions: {name: action}, or the
                                 policy robot's action alone (the others then get zeros)
    """

    def __init__(self, model, task_cls, robot_cls, options=None, seed=0):
        self.model, self.data = model, mujoco.MjData(model)
        settings = {s["name"]: s["default"] for s in list(task_cls.settings) + list(robot_cls.settings)}
        self.sim = NativeSim(model, self.data, task_cls.decimation, dict(settings, **(options or {})))
        self.task = task_cls(self.sim)
        self.names = [a["name"] for a in self.task.agents]
        self.robots = {n: robot_cls(self.sim, n) for n in self.names}
        self.rng = np.random.default_rng(seed)
        self.action_dim = robot_cls.action_dim
        self.obs = None

    def reset(self, state=None):
        sim, rng = self.sim, self.rng
        mujoco.mj_resetData(self.model, self.data)
        drawn = rng if sim.options["randomize"] else None
        for r in self.robots.values():
            r.randomize(sim, drawn)
        self.task.randomize(sim, drawn)
        self.task.reset(sim, rng, state)
        for r in self.robots.values():
            r.reset(sim, rng)
        mujoco.mj_forward(self.model, self.data)
        zeros = np.zeros(self.action_dim)
        self.pending = deque([{n: zeros for n in self.names}] * int(sim.options["action_delay"] or 0))
        self.steps = 0
        self.obs = self.observe()
        return self.obs

    def observe(self):
        return {n: self.task.observe(self.sim, n, self.rng) for n in self.names}

    def step(self, actions):
        sim, m, d = self.sim, self.model, self.data
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
            mujoco.mj_step(m, d)
        if self.task.forward_after_step:
            mujoco.mj_forward(m, d)
        self.steps += 1
        self.task.after_step(sim)
        self.obs = self.observe()
        return self.obs, self.task.done(sim)

    def metrics(self):
        out = list(self.task.metrics(self.sim))
        for r in self.robots.values():
            out += r.metrics(self.sim)
        return out


def load_classes(python_dir, entry):
    """The class at `entry` ('package.module:Class') from a bundle's python/ directory."""
    mod, _, name = entry.partition(":")
    if str(python_dir) not in sys.path:
        sys.path.insert(0, str(python_dir))
    return getattr(importlib.import_module(mod), name)


def from_bundle(sim_dir, options=None, seed=0):
    """A Host on a written bundle (sim.json, scene.mjb.gz, python/), as the extension loads it."""
    from ..bundle import load_model
    sim_dir = Path(sim_dir)
    spec = json.loads((sim_dir / "sim.json").read_text())
    py = spec["python"]
    task_cls = load_classes(sim_dir / py["dir"], py["task"])
    robot_cls = load_classes(sim_dir / py["dir"], py["robot"])
    return Host(load_model(sim_dir)[1], task_cls, robot_cls, options, seed)


__all__ = ["DEFAULT_OPTIONS", "Host", "NativeSim", "check_fields", "from_bundle", "load_classes"]
