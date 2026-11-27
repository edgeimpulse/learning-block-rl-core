# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""The simulation bundle: a copy of the simulation a policy trains in, so the RL policy simulator Studio
extension can run the policy in the browser, in the physics it trained in.

The format belongs to the extension (rl_policy_simulator_extension, whose README documents it). The robot
and the task are the block's own Python (a Robot and a Task, see ei_rl.sim), which the extension runs as-is in
Pyodide; this module checks they can run there, copies them, and writes the rest.

  <custom data>/sim/sim.json        what the extension needs before it runs the Python: timing, the agents,
                                    the commands and settings, the start states' labels, the entry points
  <custom data>/sim/python/         robot.py, env.py, what they import from the block, and ei_rl.sim
  <custom data>/sim/scene.mjb.gz    the compiled MuJoCo model the policy trained on, minus the visual-only
                                    geoms (physics_model): the physics is bitwise identical
  <custom data>/sim/visual.bin.gz   what to draw: the visible meshes, decimated (visual_model)
  <custom data>/policy/model.onnx   the trained policy
  <custom data>/checkpoints/        the policy while training, for live runs (see below)
"""

import ast
import gzip
import hashlib
import importlib.util
import inspect
import json
import shutil
import sys
import tempfile
from pathlib import Path

import mujoco
import numpy as np

FORMAT = "ei-rl-sim"
VERSION = 3
PYTHON_DIR = "python"
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

def bundle_id(spec, mjb, vis_bin, python=None):
    """A hash of what the simulation is: the physics model, the visuals, sim.json and the Python. Two
    trainings of the same task get the same id, so the extension can run a block's live checkpoints
    on another block's bundle while the first still trains."""
    h = hashlib.sha256()
    for part in (mjb, vis_bin, json.dumps({k: v for k, v in spec.items() if k != "bundle_id"}, sort_keys=True).encode(),
                 *(python[k] for k in sorted(python or {}))):
        h.update(hashlib.sha256(part).digest())
    return h.hexdigest()[:32]


def write_bundle(out_dir, spec, mjb, vis_bin, visual, python):
    """Write <out_dir>/sim/: sim.json (`spec`, plus its visual and bundle_id), scene.mjb.gz, visual.bin.gz and
    python/. Returns the sim directory.

    spec: task_spec's; visual: sim.json's visual section (visual_model's description, plus e.g.
    remote_meshes); python: python_files'.
    """
    spec["visual"] = {"file": "visual.bin.gz", **visual}
    spec["python"]["files"] = sorted(python)  # what the extension fetches into Pyodide
    spec["bundle_id"] = bundle_id(spec, mjb, vis_bin, python)
    # Written next to sim/ and swapped in: the extension may read it while training runs, and must never see
    # half a bundle, or this run's model next to a previous run's sim.json
    sim = Path(out_dir) / SIM_DIR
    tmp = sim.with_name(sim.name + ".tmp")
    shutil.rmtree(tmp, ignore_errors=True)
    tmp.mkdir(parents=True)
    (tmp / "scene.mjb.gz").write_bytes(gzip.compress(mjb, 9, mtime=0))
    (tmp / "visual.bin.gz").write_bytes(gzip.compress(vis_bin, 9, mtime=0))
    (tmp / "sim.json").write_text(json.dumps(spec, indent=1))
    for rel, src in python.items():
        (tmp / PYTHON_DIR / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp / PYTHON_DIR / rel).write_bytes(src)
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
# python/: the block's Robot and Task, as the extension runs them
# ----------------------------------------------------------------------------------------------------------
# Pyodide has the standard library and numpy, not mujoco, gymnasium or torch. So the task's and the robot's
# modules, and the block's modules they import, may import only those, ei_rl.sim, and each other. A block's
# package __init__.py usually imports more (a gymnasium registration): the bundle's are empty.

BROWSER_PACKAGES = ("numpy",)

# The standard library modules the extension's Pyodide (314.0.8, Python 3.14) can't import: those it leaves
# out, and those Python 3.13 removed (PEP 594), which the blocks' Python 3.12 still lists. Measured by
# importing every module of sys.stdlib_module_names in that Pyodide.
NOT_IN_BROWSER = frozenset({
    "curses", "dbm", "ensurepip", "grp", "idlelib", "msvcrt", "nt", "pwd", "pydoc_data", "readline", "resource",
    "syslog", "tkinter", "turtle", "turtledemo", "venv", "winreg", "winsound",
    "aifc", "audioop", "cgi", "cgitb", "chunk", "crypt", "imghdr", "lib2to3", "mailcap", "msilib", "nis",
    "nntplib", "ossaudiodev", "pipes", "sndhdr", "spwd", "sunau", "telnetlib", "uu", "xdrlib",
})


def _external_ok(base, names):
    top = base.partition(".")[0]
    if (top in sys.stdlib_module_names and top not in NOT_IN_BROWSER) or top in BROWSER_PACKAGES:
        return True
    if base == "ei_rl.sim":
        return "native" not in names  # ei_rl.sim.native is mujoco's side
    return base == "ei_rl" and set(names) == {"sim"}


def check_imports(path, module):
    """The imports of the module `module` (its file: path): [(line, imported module, [names from it])],
    relative imports resolved. `import a.b` gives ('a.b', [])."""
    tree = ast.parse(Path(path).read_text(), str(path))
    package = module if Path(path).name == "__init__.py" else module.rpartition(".")[0]
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [(node.lineno, a.name, []) for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            base = node.module or ""
            if node.level:
                parts = package.split(".")[:len(package.split(".")) - (node.level - 1)]
                base = ".".join(parts + ([node.module] if node.module else []))
            out.append((node.lineno, base, [a.name for a in node.names]))
    return out


def python_files(*classes):
    """python/ for the classes (the task and the robot): their modules, the block's modules they import (with
    empty package __init__.py files) and ei_rl.sim, as {path under python/: bytes}. Raises ValueError, with
    every offending import, if any of them imports what Pyodide can't."""
    roots = {c.__module__.partition(".")[0] for c in classes}
    files, errors, todo = {}, [], [c.__module__ for c in classes]
    for c in classes:
        if inspect.getsourcefile(c).endswith("__init__.py"):
            raise ValueError(f"{c.__qualname__} is in {c.__module__}/__init__.py, which the bundle leaves empty: "
                             "move it to a module of its own")
    while todo:
        mod = todo.pop()
        rel = mod.replace(".", "/") + ".py"
        if rel in files:
            continue
        origin = importlib.util.find_spec(mod).origin
        files[rel] = Path(origin).read_bytes()
        for line, base, names in check_imports(origin, mod):
            if _external_ok(base, names):
                continue
            where = f"{Path(origin).name}:{line}"
            if base.partition(".")[0] not in roots:
                shown = "ei_rl.sim.native" if base == "ei_rl.sim" else base
                errors.append(f"{where}: imports {shown}, which the browser doesn't have (only the standard "
                              f"library, {', '.join(BROWSER_PACKAGES)}, ei_rl.sim and the block's own modules)")
                continue
            spec = importlib.util.find_spec(base)
            if spec is None:
                errors.append(f"{where}: no module {base}")
            elif spec.submodule_search_locations is None:
                todo.append(base)
            else:  # a package: only its submodules, its __init__.py is empty in the bundle
                for n in names or [None]:
                    sub = n and importlib.util.find_spec(f"{base}.{n}")
                    if sub is None:
                        errors.append(f"{where}: {base}{'.' + n if n else ''} comes from {base}/__init__.py, which "
                                      "the bundle leaves empty: import it from its own module")
                    else:
                        todo.append(f"{base}.{n}")
    if errors:
        raise ValueError("The task's Python can't run in the extension:\n  " + "\n  ".join(errors))
    for rel in list(files):
        for i in range(1, rel.count("/") + 1):
            files.setdefault("/".join(rel.split("/")[:i]) + "/__init__.py", b"")
    sim_pkg = Path(__file__).parent / "sim"
    files["ei_rl/__init__.py"] = b'"""ei_rl.sim, the simulation contract, as the bundle ships it."""\n'
    for f in sorted(sim_pkg.glob("*.py")):
        if f.name != "native.py":
            files[f"ei_rl/sim/{f.name}"] = f.read_bytes()
    return files


def task_spec(task_cls, robot_cls, model):
    """sim.json for a task and its robot on a model (the physics model, as written): the entry points, the
    timing, what the extension's controls are, the policy's input and output sizes. Runs the task once (a
    reset and a step, without noise or randomization) to check it and to measure its observation."""
    from .sim import CONTRACT_VERSION, camera
    from .sim.native import Host, check_fields
    host = Host(model, task_cls, robot_cls, dict(noise=False, randomize=False))
    check_fields(host.model, host.data)
    obs = host.reset(0 if task_cls.states else None)
    host.step(np.zeros(robot_cls.action_dim))
    task = host.task
    dims = {n: int(np.size(o)) for n, o in obs.items()}
    entry = lambda c: f"{c.__module__}:{c.__qualname__}"  # noqa: E731
    cam = dict(camera() if task.camera is None else task.camera)
    track = cam.pop("track")
    cam["track_body_id"] = None if track is None else host.sim.id("body", track)
    for g in task.layout:
        host.sim.id("geom", g)  # each must be in the model
    return dict(
        format=FORMAT, version=VERSION,
        task=dict(id=task.id, label=task.label),
        mujoco=dict(version=mujoco.__version__, model="scene.mjb.gz", physics_dt=float(model.opt.timestep),
                    decimation=int(task.decimation)),
        python=dict(dir=PYTHON_DIR, contract=CONTRACT_VERSION, task=entry(task_cls), robot=entry(robot_cls),
                    packages=list(BROWSER_PACKAGES)),
        agents=task.agents, commands=task.commands, settings=list(task.settings) + list(robot_cls.settings),
        keys=robot_cls.keys, states=list(task.states), episode_s=task.episode_s, camera=cam, layout=list(task.layout),
        observation=dict(dim=dims[host.names[0]], dims=dims), action=dict(dim=int(robot_cls.action_dim)),
    )


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
    "BROWSER_PACKAGES", "CHECKPOINT_DIR", "FORMAT", "LIVE_FORMAT", "PYTHON_DIR", "SIM_DIR", "VERSION", "bundle_id",
    "check_imports", "checkpoint_name", "clear_policies", "cluster_mesh", "load_model", "mj_id", "model_bytes",
    "physics_model", "print_live_checkpoint", "print_live_header", "python_files", "rounded", "save_checkpoint",
    "task_spec", "uniform_noise", "visual_model", "write_bundle", "write_policy",
]
