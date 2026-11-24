# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""The simulation contract on mujoco: NativeSim, and Host, ei_rl.sim.host's loop on an MjModel. A block's
training env steps its task through a Host, so it trains in the loop the extension runs (the same code, on
MuJoCo WASM); from_bundle loads a written bundle the way the extension does, for the block's checks.
"""

import importlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from . import DATA_FIELDS, MODEL_FIELDS, MODEL_SIZES, OBJECT_KINDS, Sim, host
from .host import DEFAULT_OPTIONS

_OBJ = {k: getattr(mujoco.mjtObj, "mjOBJ_" + ("JOINT" if k == "joint" else k.upper())) for k in OBJECT_KINDS}


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

    def _reset_data(self):
        mujoco.mj_resetData(self._m, self._d)

    def _step(self):
        mujoco.mj_step(self._m, self._d)

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


class Host(host.Host):
    """host.Host on a mujoco model: Host(model, task_cls, robot_cls, options=None, seed=0). host.model and
    host.data are the MjModel and MjData."""

    def __init__(self, model, task_cls, robot_cls, options=None, seed=0):
        self.model, self.data = model, mujoco.MjData(model)
        super().__init__(NativeSim(model, self.data, task_cls.decimation), task_cls, robot_cls, options, seed)


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
