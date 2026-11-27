# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""The simulation contract on mujoco: NativeSim, and Host, ei_rl.sim.host's loop on an MjModel. A block's
training env steps its task through a Host, so it trains in the loop the extension runs (the same code, on
MuJoCo WASM); from_bundle loads a written bundle the way the extension does, for the block's checks.
"""

import functools
import importlib
import json
import sys
from pathlib import Path

import mujoco
import numpy as np

from . import OBJECT_KINDS, Sim, field_shape, host
from .host import DEFAULT_OPTIONS

_OBJ = {k: getattr(mujoco.mjtObj, "mjOBJ_" + ("JOINT" if k == "joint" else k.upper())) for k in OBJECT_KINDS}
_NUMBERS = {"mjtNum", "float", "int", "mjtByte", "mjtSize"}


def _base(t):
    while hasattr(t, "inner_type"):
        t = t.inner_type
    return getattr(t, "name", None)


@functools.cache
def field_table():
    """The contract's fields, on this mujoco: mjModel's and mjData's numeric arrays and sizes, and mjContact's
    fields, from mujoco.introspect. {'mujoco': version, 'model' and 'data': {'sizes': [names], 'arrays':
    {name: extent}}, 'contact': {name: [type, extent]}}; an extent holds sizes by name ('nmocap*3' for a
    multiple) and numbers. An array sized by something that isn't a size of the model or the data is left out."""
    from mujoco.introspect import structs

    def scalars(st):
        return [f.name for f in structs.STRUCTS[st].fields
                if type(f.type).__name__ == "ValueType" and _base(f.type) in _NUMBERS]

    enums = {m.name: int(m) for e in vars(mujoco).values() if isinstance(e, type) and e.__name__.startswith("mjt")
             for m in getattr(e, "__members__", {}).values()}

    def extent(ext):
        # MuJoCo's constants (mjNREF) and enum counts (mjNTEXROLE) as numbers; sizes stay names
        def number(e):
            v = getattr(mujoco, e, enums.get(e))
            return e if v is None else int(v)
        return [number(e) if isinstance(e, str) and e.startswith("mjN") else e for e in ext]

    sizes = {"model": scalars("mjModel"), "data": scalars("mjData")}
    known = set(sizes["model"]) | set(sizes["data"])
    table = {"mujoco": mujoco.__version__}
    for kind, st in (("model", "mjModel"), ("data", "mjData")):
        arrays = {}
        for f in structs.STRUCTS[st].fields:
            ext = getattr(f, "array_extent", None)
            if type(f.type).__name__ != "PointerType" or not ext or _base(f.type) not in _NUMBERS:
                continue
            ext = extent(ext)
            if all(not isinstance(e, str) or e.partition("*")[0] in known for e in ext):  # an unknown mjN too
                arrays[f.name] = ext
        table[kind] = {"sizes": sizes[kind], "arrays": arrays}
    table["contact"] = {f.name: [_base(f.type), list(getattr(f.type, "extents", None) or ())]
                        for f in structs.STRUCTS["mjContact"].fields if _base(f.type) in _NUMBERS}
    return table


class _Fields:
    """The contract's fields of an MjModel or MjData, and only those: what a block reads here, the browser has
    too. mujoco's arrays are views of buffers that live as long as the model or data, so an array sized by the
    model alone is looked up once; the data's sizes and what they size (nefc, efc_*) change every step."""

    def __init__(self, src, kind, model_sizes):
        t = field_table()[kind]
        object.__setattr__(self, "_src", src)
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_sizes", frozenset(t["sizes"]))
        object.__setattr__(self, "_arrays", t["arrays"])
        object.__setattr__(self, "_fixed", frozenset(
            k for k, ext in t["arrays"].items()
            if all(not isinstance(e, str) or e.partition("*")[0] in model_sizes for e in ext)))

    def __getattr__(self, k):
        if k not in self._sizes and k not in self._arrays:
            raise AttributeError(f"{k!r} is not in the simulation contract: not a numeric array or size of "
                                 f"mj{self._kind.capitalize()} (MuJoCo {mujoco.__version__})")
        v = getattr(self._src, k)
        if k in self._fixed or (self._kind == "model" and k in self._sizes):
            object.__setattr__(self, k, v)
        return v

    def __setattr__(self, k, v):
        raise AttributeError(f"assign into sim.{k}[...], not the field itself")


class _Contacts:
    """data.contact: the active contacts, each field read-only, over them."""

    def __init__(self, data):
        object.__setattr__(self, "_d", data)

    def __getattr__(self, k):
        if k not in field_table()["contact"]:
            raise AttributeError(f"{k!r} is not a field of mjContact (MuJoCo {mujoco.__version__})")
        v = np.asarray(getattr(self._d.contact, k)).view()
        v.flags.writeable = False
        return v

    def __setattr__(self, k, v):
        raise AttributeError("data.contact is read-only")


class _Data(_Fields):
    def __init__(self, src, model_sizes):
        super().__init__(src, "data", model_sizes)
        object.__setattr__(self, "contact", _Contacts(src))


class NativeSim(Sim):
    def __init__(self, model, data, decimation, options=None):
        self._m, self._d = model, data
        sizes = frozenset(field_table()["model"]["sizes"])
        self.model = _Fields(model, "model", sizes)
        self.data = _Data(data, sizes)
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
    """The table's shapes are mujoco's: the browser shapes its flat copies by them, so a mismatch would read
    garbage there."""
    t = field_table()

    def size(n):
        return getattr(model, n) if n in t["model"]["sizes"] else getattr(data, n)

    for kind, src in (("model", model), ("data", data)):
        for k, ext in t[kind]["arrays"].items():
            want, got = field_shape(ext, size), np.shape(getattr(src, k))
            if got != want:
                raise AssertionError(f"{kind}.{k}: mujoco's shape {got}, the field table's {want}")
    for k, (_, ext) in t["contact"].items():
        want, got = (data.ncon, *ext), np.shape(getattr(data.contact, k))
        if got != want:
            raise AssertionError(f"contact.{k}: mujoco's shape {got}, the field table's {want}")


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


__all__ = ["DEFAULT_OPTIONS", "Host", "NativeSim", "check_fields", "field_table", "from_bundle", "load_classes"]
