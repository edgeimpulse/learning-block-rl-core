# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""The simulation contract on MuJoCo WASM, in Pyodide: the Sim the RL policy simulator extension gives a
bundle's robot.py and env.py, and the Host that steps them (ei_rl.sim.host, the loop training runs).

MuJoCo runs in its own WASM instance, so its arrays can't be numpy views: each field is copied in when the
Python first reads it, and the data's copies are dropped (made read-only) after every mj_step, mj_forward and
mj_resetData. Copies that changed go back to MuJoCo before those, and at the end of each host call.
fields.json, next to this module, is the field table of the bundle's MuJoCo (ei_rl.sim.native.field_table):
which fields there are, and their shapes.

The extension passes `bridge`, a JS object:

  view(kind, name)     kind 'model' or 'data': MuJoCo's typed array of that field (flat), or its value (a size)
  contact(name)        that field of mjContact over the active contacts, flat (a Float64Array)
  timestep()           the model's opt.timestep
  reset_data(), step(), forward()
  name2id(kind, name), id2name(kind, id)
  ray(pnt, vec, geomgroup or null, flg_static, bodyexclude) -> [distance, geom id]
"""

import importlib
import json
from pathlib import Path

import numpy as np
from pyodide.ffi import to_js

from . import Sim, field_shape
from . import host as _host

TABLE = json.loads((Path(__file__).parent / "fields.json").read_text())
_INTS = {"int", "mjtByte", "mjtSize"}


class _Fields:
    def __init__(self, sim, kind):
        t = TABLE[kind]
        object.__setattr__(self, "_sim", sim)
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_sizes", frozenset(t["sizes"]))
        object.__setattr__(self, "_arrays", t["arrays"])

    def __getattr__(self, k):
        if k in self._sizes:
            v = self._sim._bridge.view(self._kind, k)
            if v is None:
                raise AttributeError(f"MuJoCo WASM {TABLE['mujoco']} has no {self._kind}.{k}")
            if self._kind == "model":  # the model's sizes are fixed; the data's change every step
                object.__setattr__(self, k, v)
            return v
        if k not in self._arrays:
            raise AttributeError(f"{k!r} is not in the simulation contract: not a numeric array or size of "
                                 f"mj{self._kind.capitalize()} (MuJoCo {TABLE['mujoco']})")
        return self._sim._read(self._kind, k)

    def __setattr__(self, k, v):
        raise AttributeError(f"assign into sim.{k}[...], not the field itself")


class _Contacts:
    def __init__(self, sim):
        object.__setattr__(self, "_sim", sim)

    def __getattr__(self, k):
        if k not in TABLE["contact"]:
            raise AttributeError(f"{k!r} is not a field of mjContact (MuJoCo {TABLE['mujoco']})")
        return self._sim._read_contact(k)

    def __setattr__(self, k, v):
        raise AttributeError("data.contact is read-only")


class _Data(_Fields):
    def __init__(self, sim):
        super().__init__(sim, "data")
        object.__setattr__(self, "contact", _Contacts(sim))


class BrowserSim(Sim):
    def __init__(self, bridge, decimation, options=None):
        self._bridge = bridge
        self.physics_dt = float(bridge.timestep())
        self.decimation = int(decimation)
        self.options = dict(_host.DEFAULT_OPTIONS, **(options or {}))
        self._model = {}    # name -> [array, its value as last written]
        self._data = {}     # name -> [array, its value as read], this state's
        self._contact = {}  # name -> array, this state's
        self.model = _Fields(self, "model")
        self.data = _Data(self)

    def _size(self, name):
        return getattr(self.model if name in TABLE["model"]["sizes"] else self.data, name)

    def _read(self, kind, k):
        cache = self._model if kind == "model" else self._data
        hit = cache.get(k)
        if hit is not None:
            return hit[0]
        shape = field_shape(TABLE[kind]["arrays"][k], self._size)
        v = self._bridge.view(kind, k)
        if v is None:
            raise AttributeError(f"MuJoCo WASM {TABLE['mujoco']} has no {kind}.{k}")
        a = np.asarray(v.to_py())
        if a.size != int(np.prod(shape)):
            raise RuntimeError(f"{kind}.{k}: MuJoCo WASM gives {a.size} values, the field table {shape}")
        a = a.reshape(shape)
        cache[k] = [a, a.copy()]
        return a

    def _read_contact(self, k):
        a = self._contact.get(k)
        if a is None:
            kind, ext = TABLE["contact"][k]
            flat = np.asarray(self._bridge.contact(k).to_py(), np.float64)
            a = flat.astype(np.int32) if kind in _INTS else flat
            a = a.reshape((-1, *ext))
            a.flags.writeable = False
            self._contact[k] = a
        return a

    def flush(self):
        """The Python's writes, into MuJoCo."""
        for kind, cache in (("data", self._data), ("model", self._model)):
            for k, (a, last) in cache.items():
                if not np.array_equal(a, last):
                    self._bridge.view(kind, k).assign(np.ascontiguousarray(a).reshape(-1))
                    last[...] = a

    def _drop(self):
        """Drop the data's copies, read-only first: a robot or a task that kept one (self.ctrl = sim.data.ctrl)
        then raises on its next write, rather than writing into a copy MuJoCo never sees."""
        for a, _ in self._data.values():
            a.flags.writeable = False
        self._data.clear()
        self._contact.clear()

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
    """The loop, on the extension's MuJoCo: Host(bridge, task_cls, robot_cls, options, seed). Its calls take
    and give what JS can read: observations as lists of floats, actions as sequences."""

    def __init__(self, bridge, task_cls, robot_cls, options=None, seed=0):
        sim = BrowserSim(bridge, task_cls.decimation)
        super().__init__(sim, task_cls, robot_cls, _plain(options), seed)

    def describe_js(self):
        """describe(), as JSON: JS parses it into what sim.json used to carry (None is null, not undefined)."""
        return json.dumps(self.describe())

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
    return Host(bridge, _load(spec["python"]["task"]), _load(spec["python"]["robot"]), options, seed)


def _load(entry):
    mod, _, name = entry.partition(":")
    return getattr(importlib.import_module(mod), name)


def _plain(x):
    """A JS object as Python dicts, lists and numbers."""
    return x.to_py() if hasattr(x, "to_py") else (x or {})


__all__ = ["BrowserSim", "Host", "make_host"]
