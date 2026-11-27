# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""The simulation contract on MuJoCo WASM, in Pyodide: the Sim the RL policy simulator extension gives a
bundle's robot.py and env.py, and the Host that steps them (ei_rl.sim.host, the loop training runs).

MuJoCo runs in its own WASM instance, so its arrays can't be numpy views: each field is copied in when the
Python first reads it, and the data's copies are dropped (made read-only) after every mj_step, mj_forward and
mj_resetData.
Copies go back to MuJoCo before those, and at the end of each host call: the writable data fields that were
read (WRITABLE_DATA), and the model fields that changed.

The extension passes `bridge`, a JS object:

  view(kind, name)     kind 'model' or 'data': MuJoCo's typed array of that field (flat)
  size(name)           a size of the model (nq, nbody, ...)
  time()               data.time
  reset_data(), step(), forward()
  name2id(kind, name), id2name(kind, id)
  ray(pnt, vec, geomgroup or null, flg_static, bodyexclude) -> [distance, geom id]
"""

import importlib

import numpy as np
from pyodide.ffi import to_js

from . import DATA_FIELDS, MODEL_FIELDS, MODEL_SIZES, WRITABLE_DATA, Sim
from . import host as _host


class _Fields:
    def __init__(self, sim, kind, fields, sizes=()):
        object.__setattr__(self, "_sim", sim)
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_fields", fields)
        object.__setattr__(self, "_sizes", frozenset(sizes))

    def __getattr__(self, k):
        if k in self._sizes:
            v = int(self._sim._bridge.size(k))
            object.__setattr__(self, k, v)
            return v
        if k not in self._fields:
            raise AttributeError(f"{k!r} is not in the simulation contract (ei_rl.sim MODEL_FIELDS, DATA_FIELDS)")
        return self._sim._read(self._kind, k)

    def __setattr__(self, k, v):
        raise AttributeError(f"assign into sim.{k}[...], not the field itself")


class _Data(_Fields):
    @property
    def time(self):
        return float(self._sim._bridge.time())


class BrowserSim(Sim):
    def __init__(self, bridge, physics_dt, decimation, options=None):
        self._bridge = bridge
        self.physics_dt = float(physics_dt)
        self.decimation = int(decimation)
        self.options = dict(_host.DEFAULT_OPTIONS, **(options or {}))
        self._model = {}  # name -> [array, its value as last written]
        self._data = {}   # name -> array, this state's
        self.model = _Fields(self, "model", MODEL_FIELDS, MODEL_SIZES)
        self.data = _Data(self, "data", DATA_FIELDS)

    def _shape(self, rows, cols):
        n = getattr(self.model, rows)
        return (n,) if cols is None else (n, getattr(self.model, cols) if isinstance(cols, str) else cols)

    def _read(self, kind, k):
        cache = self._model if kind == "model" else self._data
        hit = cache.get(k)
        if hit is not None:
            return hit[0] if kind == "model" else hit
        rows, cols = (MODEL_FIELDS if kind == "model" else DATA_FIELDS)[k]
        a = np.asarray(self._bridge.view(kind, k).to_py()).reshape(self._shape(rows, cols))
        if kind == "model":
            self._model[k] = [a, a.copy()]
        else:
            self._data[k] = a
        return a

    def flush(self):
        """The Python's writes, into MuJoCo."""
        for k, a in self._data.items():
            if k in WRITABLE_DATA:
                self._bridge.view("data", k).assign(np.ascontiguousarray(a).reshape(-1))
        for k, (a, last) in self._model.items():
            if not np.array_equal(a, last):
                self._bridge.view("model", k).assign(np.ascontiguousarray(a).reshape(-1))
                last[...] = a

    def _drop(self):
        """Drop the data's copies, read-only first: a robot or a task that kept one (self.ctrl = sim.data.ctrl)
        then raises on its next write, rather than writing into a copy MuJoCo never sees."""
        for a in self._data.values():
            a.flags.writeable = False
        self._data.clear()

    def _after(self, op):
        self.flush()
        op()
        self._drop()

    def _reset_data(self):
        self._after(self._bridge.reset_data)

    def _step(self):
        self._after(self._bridge.step)

    def forward(self):
        self._after(self._bridge.forward)

    def name2id(self, kind, name):
        return int(self._bridge.name2id(kind, name))

    def id2name(self, kind, i):
        n = self._bridge.id2name(kind, int(i))
        return None if n is None or n == "" else str(n)

    def ray(self, pnt, vec, geomgroup=None, flg_static=True, bodyexclude=-1):
        gg = None if geomgroup is None else to_js([int(g) for g in geomgroup])
        dist, gid = self._bridge.ray(to_js([float(v) for v in pnt]), to_js([float(v) for v in vec]), gg,
                                     int(bool(flg_static)), int(bodyexclude)).to_py()
        return float(dist), int(gid)


class Host(_host.Host):
    """The loop, on the extension's MuJoCo: Host(bridge, spec, task_cls, robot_cls, options, seed). Its calls
    take and give what JS can read: observations as lists of floats, actions as sequences."""

    def __init__(self, bridge, spec, task_cls, robot_cls, options=None, seed=0):
        sim = BrowserSim(bridge, spec["mujoco"]["physics_dt"], task_cls.decimation)
        super().__init__(sim, task_cls, robot_cls, _plain(options), seed)

    def reset_js(self, state=None):
        self.sim._drop()  # the page may have moved things since the last call
        obs = self.reset(None if state is None else int(state))
        self.sim.flush()
        return {n: o.tolist() for n, o in obs.items()}

    def step_js(self, actions):
        self.sim._drop()
        acts = {str(n): np.asarray(list(a), np.float64) for n, a in _plain(actions).items()}
        obs, d = self.step(acts)
        self.sim.flush()
        return {n: o.tolist() for n, o in obs.items()}, d

    def set_options(self, options):
        o = _plain(options)
        if "seed" in o:
            self.rng = np.random.default_rng(int(o.pop("seed")))
        self.sim.options.update(o)

    def set_command(self, name, value):
        self.sim._drop()
        self.task.set_command(str(name), np.asarray(list(value), np.float64))
        self.sim.flush()

    def commands(self):
        return {n: v.tolist() for n, v in self.task.command_values.items()}

    def panel(self):
        """What the extension shows after a step: {'metrics': [...], 'markers': [...]}."""
        self.sim._drop()
        return dict(metrics=self.metrics(), markers=self.markers())


def make_host(bridge, spec, options=None, seed=0):
    """The Host of a bundle (its sim.json, as JS gives it), on the extension's MuJoCo."""
    spec = _plain(spec)
    return Host(bridge, spec, _load(spec["python"]["task"]), _load(spec["python"]["robot"]), options, seed)


def _load(entry):
    mod, _, name = entry.partition(":")
    return getattr(importlib.import_module(mod), name)


def _plain(x):
    """A JS object as Python dicts, lists and numbers."""
    return x.to_py() if hasattr(x, "to_py") else (x or {})


__all__ = ["BrowserSim", "Host", "make_host"]
