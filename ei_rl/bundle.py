# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""The simulation bundle: a copy of the simulation a policy trains in, so the RL policy simulator Studio
extension can run the policy in the browser, in the physics it trained in.

The format belongs to the extension (rl_policy_simulator_extension, whose README documents it). This module
writes its generic parts; each block describes its own task (sim.json's robot, joints, action, actuator,
observation, terminations, randomization, initial states) and keeps a reference engine to check it.

  <custom data>/sim/sim.json        the robot and the task: what the policy observes, how its actions drive
                                    the motors, the commands, the start states (see the extension's README)
  <custom data>/sim/scene.mjb.gz    the compiled MuJoCo model the policy trained on, minus the visual-only
                                    geoms (physics_model): the physics is bitwise identical
  <custom data>/sim/visual.bin.gz   what to draw: the visible meshes, decimated (visual_model)
  <custom data>/policy/model.onnx   the trained policy
  <custom data>/checkpoints/        the policy while training, for live runs (see below)
"""

import gzip
import hashlib
import json
import shutil
import tempfile
from pathlib import Path

import mujoco
import numpy as np

FORMAT = "ei-rl-sim"
VERSION = 1
SIM_DIR = "sim"
POLICY_DIR = "policy"


# ----------------------------------------------------------------------------------------------------------
# sim.json helpers
# ----------------------------------------------------------------------------------------------------------

def rounded(x, digits=7):
    return [round(float(v), digits) for v in x]


def mj_id(model, kind, name):
    i = mujoco.mj_name2id(model, kind, name)
    if i < 0:
        raise KeyError(f"{kind.name} {name!r} not in the model")
    return i


def uniform_noise(half):
    """An observation term's noise: uniform in [-half, half], added."""
    return {"type": "uniform", "min": -half, "max": half, "operation": "add"}


# ----------------------------------------------------------------------------------------------------------
# scene.mjb.gz: the physics
# ----------------------------------------------------------------------------------------------------------

CHECKED_SIZES = ("nq", "nv", "nu", "nbody", "njnt", "nsite", "nmocap", "nkey")
CHECKED_FIELDS = ("body_mass", "body_inertia", "body_ipos", "body_iquat", "dof_armature", "dof_damping",
                  "dof_frictionloss", "jnt_range", "actuator_gear", "actuator_ctrlrange", "actuator_forcerange",
                  "actuator_gainprm", "actuator_biasprm", "qpos0", "site_pos", "body_mocapid", "key_qpos")


def model_bytes(model):
    buf = np.zeros(mujoco.mj_sizeModel(model), np.uint8)
    mujoco.mj_saveModel(model, None, buf)
    return buf.tobytes()


def physics_model(full, spec, reset=None, control=None, hold=10, steps=1000, fix_inertia=True, after_compile=None):
    """`full` without its visual-only geoms, and the bytes of its MJB.

    spec: the MjSpec `full` was compiled from (it is modified). Deletes the geoms that collide with nothing and
    the meshes no geom uses any more, and compiles. Bodies, joints, sites and actuators keep their ids; geoms
    don't. Then checks the dynamics are unchanged: the same sizes, inertias, joint and actuator parameters, and
    the same trajectory over `steps` physics steps under random controls.

    reset(model, data): the start state of the check (default: mj_resetData).
    control(rng, model) -> ctrl: a random control, held for `hold` steps (default: uniform in the control
        ranges, or [-1, 1] for an unlimited actuator).
    fix_inertia: first set each body's inertial properties to the full model's, since some visual geoms carry
        mass; leave it off for models whose visual geoms have none (to keep the MJB as the scene compiles).
    after_compile(model): what the env changes on its compiled model (e.g. joint damping), applied here too.
    """
    if fix_inertia:
        for b in range(1, full.nbody):
            body = spec.body(full.body(b).name)
            body.explicitinertial = True
            body.mass = full.body_mass[b]
            body.ipos = full.body_ipos[b]
            body.iquat = full.body_iquat[b]
            body.inertia = full.body_inertia[b]
            body.fullinertia = [np.nan] * 6  # unset: the diagonal one above, in the iquat frame
    for g in [g for g in spec.geoms if g.contype == 0 and g.conaffinity == 0]:
        spec.delete(g)
    used = {g.meshname for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_MESH}
    for msh in list(spec.meshes):
        if (msh.name or Path(msh.file).stem) not in used:  # an unnamed mesh is named after its file
            spec.delete(msh)
    m = spec.compile()
    if after_compile is not None:
        after_compile(m)

    bad = [s for s in CHECKED_SIZES if getattr(m, s) != getattr(full, s)]
    for f in CHECKED_FIELDS:
        if not bad and not np.array_equal(getattr(full, f), getattr(m, f)):
            bad.append(f)
    drift = 0.0
    if not bad:
        reset = reset or (lambda model, data: mujoco.mj_resetData(model, data))
        if control is None:
            lo, hi = full.actuator_ctrlrange.T.copy()
            free = ~full.actuator_ctrllimited.astype(bool)
            lo[free], hi[free] = -1.0, 1.0
            control = lambda rng, model: rng.uniform(lo, hi)  # noqa: E731
        d1, d2 = mujoco.MjData(full), mujoco.MjData(m)
        for d, mm in ((d1, full), (d2, m)):
            reset(mm, d)
            mujoco.mj_forward(mm, d)
        rng = np.random.default_rng(0)
        for k in range(steps):
            if k % hold == 0:
                u = control(rng, full)
            d1.ctrl[:] = u
            d2.ctrl[:] = u
            mujoco.mj_step(full, d1)
            mujoco.mj_step(m, d2)
        drift = float(np.abs(d1.qpos - d2.qpos).max())
    if bad or drift > 1e-9:
        raise RuntimeError(f"stripped physics model differs from the trained one: {bad}, drift {drift:g}")
    return m, model_bytes(m)


def load_model(sim_dir):
    """The bundle's physics model (scene.mjb.gz), for a reference engine."""
    sim_dir = Path(sim_dir)
    spec = json.loads((sim_dir / "sim.json").read_text())
    with tempfile.NamedTemporaryFile(suffix=".mjb") as f:
        f.write(gzip.decompress((sim_dir / spec["mujoco"]["model"]).read_bytes()))
        f.flush()
        return spec, mujoco.MjModel.from_binary_path(f.name)


# ----------------------------------------------------------------------------------------------------------
# visual.bin.gz: what to draw
# ----------------------------------------------------------------------------------------------------------

def cluster_mesh(v, f, cell):
    """Vertex-clustering decimation: snap vertices to a `cell` grid, merge each cell, drop degenerate faces."""
    keys = np.floor(v / cell).astype(np.int64)
    _, inv = np.unique(keys, axis=0, return_inverse=True)
    inv = inv.ravel()
    n = inv.max() + 1
    nv = np.zeros((n, 3))
    np.add.at(nv, inv, v)
    nv /= np.bincount(inv, minlength=n)[:, None]
    nf = inv[f]
    keep = (nf[:, 0] != nf[:, 1]) & (nf[:, 1] != nf[:, 2]) & (nf[:, 0] != nf[:, 2])
    nf = nf[keep]
    if len(nf):
        nf = np.unique(nf, axis=0)  # faces that collapsed onto the same triangle
    used, remap = np.unique(nf, return_inverse=True)
    if len(used) < 3:  # a mesh smaller than a cell: keep it as is
        return v.astype(np.float32), f.astype(np.uint32)
    return nv[used].astype(np.float32), remap.reshape(-1, 3).astype(np.uint32)


def visual_model(full, cell=0.001, groups=(0, 1, 2), with_group=False):
    """What to draw: every geom in a visible group, each mesh once, decimated to a `cell` grid (m).

    Mesh vertices are in the mesh frame of the compiled model, so geom_pos / geom_quat place them exactly.
    with_group: give each geom its group (for sim.json's visual.remote_meshes, which replace a group).
    Returns (bytes of visual.bin, its description for sim.json: meshes and geoms).
    """
    mesh_index, meshes, chunks, offset = {}, [], [], 0
    geoms = []
    for g in range(full.ngeom):
        if full.geom_group[g] not in groups:
            continue
        mat = int(full.geom_matid[g])
        rgba = full.mat_rgba[mat] if mat >= 0 else full.geom_rgba[g]
        body = int(full.geom_bodyid[g])
        entry = {
            "name": mujoco.mj_id2name(full, mujoco.mjtObj.mjOBJ_GEOM, g) or "",
            "body": mujoco.mj_id2name(full, mujoco.mjtObj.mjOBJ_BODY, body) or "",
            "body_id": body,
            "pos": rounded(full.geom_pos[g]),
            "quat": rounded(full.geom_quat[g]),
            "rgba": [round(float(x), 4) for x in rgba],
        }
        t = int(full.geom_type[g])
        if t == mujoco.mjtGeom.mjGEOM_MESH:
            k = int(full.geom_dataid[g])
            if k not in mesh_index:
                va, nvert = full.mesh_vertadr[k], full.mesh_vertnum[k]
                fa, nface = full.mesh_faceadr[k], full.mesh_facenum[k]
                v, f = cluster_mesh(full.mesh_vert[va:va + nvert].astype(np.float64),
                                    full.mesh_face[fa:fa + nface], cell)
                mesh_index[k] = len(meshes)
                meshes.append({"name": mujoco.mj_id2name(full, mujoco.mjtObj.mjOBJ_MESH, k) or "",
                               "vert_offset": offset, "vert_count": len(v),
                               "face_offset": offset + v.nbytes, "face_count": len(f)})
                chunks += [v.tobytes(), f.tobytes()]
                offset += v.nbytes + f.nbytes
            entry["mesh"] = mesh_index[k]
        else:
            entry["type"] = mujoco.mjtGeom(t).name.removeprefix("mjGEOM_").lower()
            entry["size"] = rounded(full.geom_size[g])
        if with_group:
            entry["group"] = int(full.geom_group[g])
        geoms.append(entry)
    return b"".join(chunks), {"meshes": meshes, "geoms": geoms}


# ----------------------------------------------------------------------------------------------------------
# The bundle
# ----------------------------------------------------------------------------------------------------------

def bundle_id(spec, mjb, vis_bin):
    """A hash of what the simulation is: the physics model, the visuals and sim.json without its sampled start
    states. Two trainings of the same task get the same id, so the extension can run a block's live checkpoints
    on another block's bundle while the first still trains."""
    h = hashlib.sha256()
    for part in (mjb, vis_bin, json.dumps({k: v for k, v in spec.items() if k not in ("initial_states", "bundle_id")},
                                          sort_keys=True).encode()):
        h.update(hashlib.sha256(part).digest())
    return h.hexdigest()[:32]


def write_bundle(out_dir, spec, mjb, vis_bin, visual):
    """Write <out_dir>/sim/: sim.json (`spec`, plus its visual and bundle_id), scene.mjb.gz, visual.bin.gz.
    Returns the sim directory.

    visual: sim.json's visual section (visual_model's description, plus e.g. remote_meshes).
    """
    spec["visual"] = {"file": "visual.bin.gz", **visual}
    spec["bundle_id"] = bundle_id(spec, mjb, vis_bin)
    # Written next to sim/ and swapped in: the extension may read it while training runs, and must never see
    # half a bundle, or this run's model next to a previous run's sim.json
    sim = Path(out_dir) / SIM_DIR
    tmp = sim.with_name(sim.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    (tmp / "scene.mjb.gz").write_bytes(gzip.compress(mjb, 9, mtime=0))
    (tmp / "visual.bin.gz").write_bytes(gzip.compress(vis_bin, 9, mtime=0))
    (tmp / "sim.json").write_text(json.dumps(spec, indent=1))
    shutil.rmtree(sim, ignore_errors=True)
    tmp.rename(sim)
    faces = sum(m["face_count"] for m in visual.get("meshes", []))
    print(f"Simulation bundle: {sim} (physics {len(mjb) / 1e3:.0f} kB, {len(visual.get('geoms', []))} visual geoms, "
          f"{faces} triangles, {sum(f.stat().st_size for f in sim.iterdir()) / 1e3:.0f} kB on disk)")
    return sim


def write_policy(out_dir, onnx_path):
    """<out_dir>/policy/model.onnx: the trained policy, next to sim/."""
    d = Path(out_dir) / POLICY_DIR
    d.mkdir(parents=True, exist_ok=True)
    shutil.copy2(onnx_path, d / "model.onnx")
    return d / "model.onnx"


# ----------------------------------------------------------------------------------------------------------
# Live checkpoints: policies the extension can run while the job still trains
# ----------------------------------------------------------------------------------------------------------
# The custom data directory is on a network share that Studio's custom-data API reads, so the extension reads
# sim/ and the checkpoints as soon as they are written. The log only says which files belong to this run (a
# previous run's may still be there): `[rl-sim-run]` once sim/ is written, then one `[rl-sim-checkpoint]` line
# per checkpoint once its file is complete. Studio's log download keeps only the last 10,000 lines (Stable-Baselines3
# alone logs ~25 per iteration), so each checkpoint line comes after the run's line again.

LIVE_FORMAT = "ei-rl-sim-live"
LIVE_RUN_PREFIX = "[rl-sim-run]"
LIVE_CHECKPOINT_PREFIX = "[rl-sim-checkpoint]"
CHECKPOINT_DIR = "checkpoints"
_live_header = None  # the last `[rl-sim-run]` line, printed again with each checkpoint


def clear_policies(out_dir):
    """Remove a previous run's checkpoints/ and policy/: the extension would list them with this run's."""
    for d in (CHECKPOINT_DIR, POLICY_DIR):
        shutil.rmtree(Path(out_dir) / d, ignore_errors=True)


def print_live_header(sim_dir):
    """`[rl-sim-run]`: the simulation this run's checkpoints run on (sim/ in the custom data)."""
    spec = json.loads((Path(sim_dir) / "sim.json").read_text())
    header = dict(format=LIVE_FORMAT, version=VERSION, task=spec["task"]["id"], bundle_id=spec["bundle_id"],
                  mujoco=spec["mujoco"]["version"], obs_dim=spec["observation"]["dim"],
                  action_dim=spec["action"]["dim"], sim_dir=Path(sim_dir).name, checkpoint_dir=CHECKPOINT_DIR)
    global _live_header
    _live_header = f"{LIVE_RUN_PREFIX} " + json.dumps(header, separators=(",", ":"))
    print(_live_header, flush=True)


def checkpoint_name(step):
    return f"{CHECKPOINT_DIR}/step_{step:010d}.onnx"


def print_live_checkpoint(path, step, iteration, label=None):
    """`[rl-sim-checkpoint]`: a checkpoint is in the custom data, at checkpoint_name(step). After the run's
    `[rl-sim-run]` line again, which a long log's last 10,000 lines would otherwise lose."""
    if _live_header is not None:
        print(_live_header, flush=True)
    line = dict(step=step, iteration=iteration, label=label, path=checkpoint_name(step), bytes=Path(path).stat().st_size)
    print(f"{LIVE_CHECKPOINT_PREFIX} " + json.dumps(line, separators=(",", ":")), flush=True)


def save_checkpoint(out_dir, step, iteration, label, export, live=True):
    """Export the current policy to <out_dir>/checkpoints/step_<n>.onnx and announce it.

    export(path): writes the ONNX model to `path`. Written to a temporary name first: the extension may read
    the file as soon as it appears. live: print `[rl-sim-checkpoint]` (only if `[rl-sim-run]` was printed).
    """
    path = Path(out_dir) / checkpoint_name(step)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".onnx.tmp")
    export(tmp)
    tmp.replace(path)
    if live:
        print_live_checkpoint(path, step, iteration, label)
    print(f"Checkpoint at iteration {iteration}: {path}", flush=True)
    return path


__all__ = [
    "CHECKPOINT_DIR", "FORMAT", "LIVE_FORMAT", "SIM_DIR", "VERSION", "bundle_id", "checkpoint_name", "clear_policies",
    "cluster_mesh", "load_model", "mj_id", "model_bytes", "physics_model", "print_live_checkpoint",
    "print_live_header", "rounded", "save_checkpoint", "uniform_noise", "visual_model", "write_bundle",
    "write_policy",
]


# ----------------------------------------------------------------------------------------------------------
# The observer hook (format version 3): the robot's own observation code, as ONNX
# ----------------------------------------------------------------------------------------------------------
# Some robots observe through their own code: a complementary filter on an IMU, a low-passed command, a leaky
# integrator of the heading. That is state over time, which no observation term kind describes, so the block
# ships the code: a torch module, exported to ONNX, that the extension runs at every policy step (see its
# README, "Observer hook"). The module's forward takes, each float32 [1, n]:
#   sensors      the MuJoCo sensors sim.json's observer names, concatenated, read after the step's physics
#   last_action  the policy's last action after the action clip (zeros at reset), before the action delay
#   state        its own state_out from the last step (zeros at reset)
#   noise        unit draws (normal: std 1, uniform: in [-1, 1]), zeros with noise off: the graph scales them
#   reset        1 on the call after a reset, else 0
# and returns obs and state_out. Falls stay in sim.json's terminations (for one about a trim: a framezaxis
# sensor on a turned site, in terminations.sensors). Train with the same module
# (or check the environment against ObserverHook), so the bundle runs the code the policy learned on.

OBSERVER_INPUTS = ("sensors", "last_action", "state", "noise", "reset")
OBSERVER_NOISE = ("normal", "uniform")
OBSERVER_OPSET = 17
# Ops the observer graph must not have: the page draws the noise
RANDOM_OPS = {"RandomNormal", "RandomNormalLike", "RandomUniform", "RandomUniformLike", "Multinomial", "Bernoulli"}


def atan2(y, x):
    """torch.atan2 for an ONNX graph. torch exports torch.atan2 as Atan(y / x) and corrections that go wrong at
    x = -0 (atan2(1, -0) gives -pi/2, not pi/2), which a negated sensor reading of exactly 0 is: an observer
    computing atan2(acc_z, -acc_y) hits it. This takes Atan of y / |x|, then the quadrant from x < 0 (-0 counts
    as 0, in y too: atan2(-0, -1) is pi, where numpy says -pi), and 0 at (0, 0). Otherwise within 2.4e-7 of numpy's."""
    import math

    import torch
    ax = torch.abs(x)
    base = torch.atan(y / torch.where((ax == 0) & (y == 0), torch.ones_like(ax), ax))
    return torch.where(x < 0, torch.where(y >= 0, math.pi - base, -math.pi - base), base)


def observer_spec(model, file, sensors, state_dim, noise, obs_dim):
    """sim.json's observer section. sensors: MuJoCo sensor names, in the order the module reads them;
    noise: [{"dist": "normal" | "uniform", "dim", "label"}], the unit draws in the order the module takes them."""
    for n in sensors:
        mj_id(model, mujoco.mjtObj.mjOBJ_SENSOR, n)
    for n in noise:
        if n["dist"] not in OBSERVER_NOISE:
            raise ValueError(f"observer noise {n['dist']!r}: the extension draws {OBSERVER_NOISE}")
    return {"file": file, "sensors": list(sensors), "state_dim": int(state_dim),
            "noise_dim": int(sum(n["dim"] for n in noise)), "noise": [dict(n) for n in noise], "obs_dim": int(obs_dim)}


def sensor_dim(model, sensors):
    return int(sum(model.sensor(n).dim[0] for n in sensors))


def export_observer(module, path, sensor_dim, action_dim, state_dim, noise_dim, samples=256, seed=0, atol=1e-5):
    """Export the observer module to ONNX at `path`, as the extension runs it, and check the graph: no random
    ops, and onnxruntime gives what torch gives on random inputs, with some sensors at 0 and -0 (which catches
    an exported torch.atan2: use ei_rl.bundle.atan2). Returns the path."""
    import onnx
    import onnxruntime as ort
    import torch

    dims = dict(zip(OBSERVER_INPUTS, (sensor_dim, action_dim, state_dim, noise_dim, 1)))
    outputs = ["obs", "state_out"]
    module = module.eval()
    zeros = tuple(torch.zeros(1, d) for d in dims.values())
    path = Path(path)
    torch.onnx.export(module, zeros, str(path), opset_version=OBSERVER_OPSET, do_constant_folding=True,
                      dynamo=False, external_data=False, input_names=list(OBSERVER_INPUTS), output_names=outputs)
    graph = onnx.load(str(path))
    bad = sorted({n.op_type for n in graph.graph.node} & RANDOM_OPS)
    if bad:
        raise ValueError(f"{path.name}: the observer graph has {bad}. Draw noise from the noise input")
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    rng = np.random.default_rng(seed)
    for i in range(samples):
        feeds = {k: rng.normal(0, 1, (1, d)).astype(np.float32) for k, d in dims.items()}
        feeds["reset"][:] = float(i % 8 == 0)
        if i % 4 == 1:  # readings of exactly 0 and -0: where torch's atan2 export goes wrong
            feeds["sensors"][0, rng.random(sensor_dim) < 0.5] = 0.0
            feeds["sensors"][0, rng.random(sensor_dim) < 0.5] = -0.0
        with torch.no_grad():
            want = module(*(torch.from_numpy(feeds[k]) for k in OBSERVER_INPUTS))
        got = session.run(outputs, feeds)
        for name, w, g in zip(outputs, want, got):
            w = w.numpy()
            if g.shape != w.shape or not np.allclose(g, w, atol=atol, rtol=1e-5):
                raise ValueError(f"{path.name}: ONNX {name} differs from torch (max {np.max(np.abs(g - w)):.3g}) on "
                                 f"{ {k: v.tolist() for k, v in feeds.items()} }")
    return path


class ObserverHook:
    """The observer graph as the extension runs it, for a block's reference engine and its --check: reset() at an
    episode's start (after the start state is set and forwarded), step() after each policy step's physics.
    Both read the sensors from `data` and return the observation."""

    def __init__(self, model, spec, path):
        import onnxruntime as ort
        self.spec = spec
        self.session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self.outputs = ["obs", "state_out"]
        self.slices = [slice(model.sensor(n).adr[0], model.sensor(n).adr[0] + model.sensor(n).dim[0])
                       for n in spec["sensors"]]
        self.state = np.zeros(spec["state_dim"], np.float32)

    def noise(self, rng):
        """Unit draws as the extension makes them (any rng with normal and uniform)."""
        return np.concatenate([rng.normal(0, 1, n["dim"]) if n["dist"] == "normal" else rng.uniform(-1, 1, n["dim"])
                               for n in self.spec["noise"]] or [np.zeros(0)]).astype(np.float32)

    def reset(self, data, action_dim, noise=None):
        self.state = np.zeros(self.spec["state_dim"], np.float32)
        return self._run(data, np.zeros(action_dim), noise, 1.0)

    def step(self, data, last_action, noise=None):
        return self._run(data, last_action, noise, 0.0)

    def _run(self, data, last_action, noise, reset):
        sd = data.sensordata
        feeds = {"sensors": np.concatenate([sd[s] for s in self.slices]).astype(np.float32)[None],
                 "last_action": np.asarray(last_action, np.float32)[None],
                 "state": self.state[None],
                 "noise": (np.zeros(self.spec["noise_dim"], np.float32) if noise is None
                           else np.asarray(noise, np.float32))[None],
                 "reset": np.array([[reset]], np.float32)}
        out = dict(zip(self.outputs, self.session.run(self.outputs, feeds)))
        self.state = out["state_out"][0]
        obs = out["obs"][0]
        if obs.shape[0] != self.spec["obs_dim"]:
            raise ValueError(f"the observer returned {obs.shape[0]} values, sim.json says obs_dim {self.spec['obs_dim']}")
        return obs
