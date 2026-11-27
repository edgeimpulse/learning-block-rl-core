# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""The simulation contract: a block's own Python, which trains in the block and runs as-is in the RL policy
simulator extension (in Pyodide, next to MuJoCo WASM).

A block describes its robot and its task in two files:

  robot.py   a Robot: how one robot's actions drive its motors (ctrl or applied forces), every physics step
  env.py     a Task: the start states, what the robots observe, when an episode ends, and what the
             extension shows (commands, settings, metrics, markers)

Both run against a Sim: the MuJoCo model and data as numpy arrays (MuJoCo's own fields, see below), and a few
functions. The same Sim runs in training
(ei_rl.sim.native, on mujoco) and in the extension (ei_rl.sim.browser, on MuJoCo WASM). So the two files
import only the standard library, numpy, ei_rl.sim and each other; ei_rl.bundle checks that and copies them,
with this module, into the bundle. This module itself imports nothing else.

A host steps them: ei_rl.sim.native.Host in training and in the block's checks, the extension's engine in the
browser. It makes one Robot per agent of the task, Robot(sim, name), then the Task, Task(sim, robots): robots
is {agent name: Robot}, task.robots, so the task can read its robots (their last action, their randomization).
Each episode and each policy step, in this order:

  reset(state)    mj_resetData
                  robot.randomize(sim, rng) for each robot, then task.randomize(sim, rng): rng None when
                    randomization is off, for the nominal values
                  task.reset(sim, rng, state)
                  robot.reset(sim, rng) for each robot
                  mj_forward
                  task.observe(sim, name, rng) for each robot

  step(actions)   robot.command(sim, action) for each robot (the action of action_delay steps ago)
                  decimation times: robot.apply(sim) for each robot, then mj_step
                  mj_forward (unless task.forward_after_step is False)
                  task.after_step(sim)
                  task.observe(sim, name, rng) for each robot (given obs_delay steps later)
                  task.done(sim)

rng is a numpy Generator, seeded by the host: the same seed gives the same random draws in training and in
the browser (and the same episode when their MuJoCo versions match). sim.options holds the extension's
switches (noise, randomize, obs_delay, action_delay) and the values of the declared settings.
"""

import numpy as np

CONTRACT_VERSION = 2

# The fields a Sim gives are MuJoCo's own: every numeric array and size of mjModel and mjData, as
# mujoco.introspect lists them for the bundle's MuJoCo version (ei_rl.sim.native.field_table; the bundle ships
# it as ei_rl/sim/fields.json), shaped as the mujoco Python bindings shape them. A field MuJoCo adds is in the
# contract on the next MuJoCo version, with no change here or in the extension.
# The sizes of mjData (ncon, nefc, ...) change every step, and so do the arrays they size (efc_*): read them
# each time. data.contact: the active contacts, read-only, each field an array over them (contact.geom: ncon × 2,
# contact.dist: ncon).
# Natively these are mujoco's own arrays; in the browser, copies, written back after each call if they changed.
# So don't keep a data field across calls: read it from sim.data each time (a kept copy is read-only).
# Not in it: the structs other than contact (opt, vis, stat, the solver's statistics), strings (names: name2id,
# id2name), and MuJoCo's functions other than mj_step, mj_forward, mj_resetData and mj_ray.


def field_shape(extent, size):
    """A field's shape: its extent in the field table (sizes by name, 'nmocap*3', numbers), size(name) a size."""
    dims = []
    for e in extent:
        if isinstance(e, str):
            name, _, k = e.partition("*")
            e = size(name) * (int(k) if k else 1)
        dims.append(int(e))
    return tuple(dims)


OBJECT_KINDS = ("body", "joint", "geom", "site", "camera", "actuator", "sensor", "mesh", "tendon")


class Sim:
    """What robot.py and env.py see. Implemented by ei_rl.sim.native.NativeSim and ei_rl.sim.browser.

    model, data   mjModel's and mjData's numeric arrays and sizes, as numpy arrays and numbers (data.time, the
                  simulation time, s); data.contact, the active contacts
    physics_dt    the physics time step (s); decimation: physics steps per policy step; dt: the policy step
    options       the host's switches and the settings' values (see the module docstring)
    """

    physics_dt = None
    decimation = None
    options = None

    @property
    def dt(self):
        return self.physics_dt * self.decimation

    def name2id(self, kind, name):
        """The id of a named object (kind: one of OBJECT_KINDS), -1 if there is none."""
        raise NotImplementedError

    def id2name(self, kind, i):
        """The name of an object, None if it has none."""
        raise NotImplementedError

    def ray(self, pnt, vec, geomgroup=None, flg_static=True, bodyexclude=-1):
        """mj_ray: (the distance along vec from pnt to the first geom it hits, that geom's id); (-1, -1) if
        none. geomgroup: 6 flags, the geom groups to test (None: all)."""
        raise NotImplementedError

    def forward(self):
        """mj_forward: the positions, sensors and forces of the current state."""
        raise NotImplementedError

    def id(self, kind, name):
        """name2id, but a missing name is an error."""
        i = self.name2id(kind, name)
        if i < 0:
            raise KeyError(f"{kind} {name!r} not in the model")
        return i


# ----------------------------------------------------------------------------------------------------------
# What a task and a robot declare for the extension: plain dicts, which Host.describe passes on as they are
# ----------------------------------------------------------------------------------------------------------

def agent(name, label=None, drivers=("policy", "keyboard", "none")):
    """A robot in the task. The task's first is the policy's; each other one is driven by one of `drivers`
    (the trained policy on its own observation, the keyboard through Robot.keys, or nothing), picked in the
    extension."""
    return dict(name=name, label=label or name.replace("_", " ").capitalize(), drivers=list(drivers))


def command(name, dim, low, high, default=None, label=None, axes=None, keys=None, widget="slider"):
    """Something the user sets while the policy runs, and the policy observes (a velocity to track, a
    target). widget: 'slider' (one per axis), 'buttons' (low, high and default as presets) or 'point' (a
    point on the ground, picked in the scene: x, y). keys: {key: [axis, value]}, keyboard shortcuts."""
    default = [0.0] * dim if default is None else [float(v) for v in np.broadcast_to(default, dim)]
    return dict(name=name, label=label or name.replace("_", " ").capitalize(), dim=dim, widget=widget,
                low=[float(v) for v in np.broadcast_to(low, dim)], high=[float(v) for v in np.broadcast_to(high, dim)],
                default=default, axes=list(axes) if axes else [str(i) for i in range(dim)], keys=keys or {})


def setting(name, low, high, default, label=None, unit=None, step=None):
    """A value of the simulation the user sets (a battery voltage, a payload), read from sim.options[name] at
    the next reset; the policy doesn't observe it."""
    return dict(name=name, label=label or name.replace("_", " ").capitalize(), low=float(low), high=float(high),
                default=float(default), unit=unit, step=step)


def done(reason, label=None, tone="info", **info):
    """What Task.done returns when the episode ends. reason: a short word ('time_out', 'fell', 'goal', 'lost');
    label: what the extension shows ('Blue scored'); tone: 'success', 'failure' or 'info'. The extension counts
    the episodes by reason."""
    return dict(reason=reason, label=label or reason.replace("_", " ").capitalize(), tone=tone, **info)


def metric(label, value, unit=None, chart=True, low=None):
    """One of the extension's live readings. chart: plot it over the episode; low: the chart's floor."""
    return dict(label=label, value=float(value), unit=unit, chart=chart, low=low)


def camera(distance=1.0, elevation_deg=-20.0, azimuth_deg=135.0, lookat=(0.0, 0.0, 0.1), track=None):
    """Where the extension's viewer starts: `distance` (m) from what it looks at, `elevation_deg` below the
    horizon (< 0), `azimuth_deg` around z from +x (MuJoCo's convention). It looks at `lookat` (world, m), or
    follows the body named `track` (the user can turn following off)."""
    return dict(distance=float(distance), elevation_deg=float(elevation_deg), azimuth_deg=float(azimuth_deg),
                lookat=[float(v) for v in lookat], track=track)


def arrow(pos, vec, rgba=(1.0, 0.8, 0.1, 1.0), label=None):
    """A marker: an arrow from pos (world, m) along vec."""
    return dict(kind="arrow", pos=[float(v) for v in pos], vec=[float(v) for v in vec], rgba=list(rgba), label=label)


def point(pos, rgba=(1.0, 0.8, 0.1, 1.0), radius=0.02, label=None):
    """A marker: a sphere at pos (world, m)."""
    return dict(kind="point", pos=[float(v) for v in pos], rgba=list(rgba), radius=radius, label=label)


# ----------------------------------------------------------------------------------------------------------
# The two classes a block implements
# ----------------------------------------------------------------------------------------------------------

class Robot:
    """How one robot's actions drive its motors. The host makes one per agent of the task:
    Robot(sim, name). Store the ids and addresses it needs in __init__ (sim.id(...)).

    action_dim   the length of its action
    keys         {key: [action index, value]}: how the keyboard drives it (held keys add up)
    settings     setting(...)s: values the user sets, in sim.options at reset
    """

    action_dim = None
    keys = {}
    settings = []

    def __init__(self, sim, name):
        self.name = name

    def randomize(self, sim, rng):
        """Draw this robot's physical parameters (motor constants, masses): write absolute values, from the
        nominal ones kept in __init__. rng None: the nominal values."""

    def reset(self, sim, rng):
        """A new episode: clear the actuator state (filters, delays, the last command)."""

    def command(self, sim, action):
        """A new action, once per policy step."""

    def apply(self, sim):
        """Before every physics step: write sim.data.ctrl (or qfrc_applied, xfrc_applied) from the command."""

    def metrics(self, sim):
        """metric(...)s for the extension's panel, after each policy step."""
        return []


class Task:
    """What the robots do: the scene's start states, the observations, the end of an episode, and what the
    extension shows. The host makes one, after the robots: Task(sim, robots), robots {agent name: Robot} (self.robots).

    id, label           the task's id (as in the training's task descriptor) and a readable name
    agents              agent(...)s; the first is the policy's. A class attribute: the host reads it before it makes the task
    decimation          physics steps per policy step
    episode_s           the episode length (s), for the default done(); None: no time limit
    commands, settings  command(...)s and setting(...)s, see those
    states              labels of the start states the extension offers (reset's `state` is an index into it;
                        None: the task draws one)
    forward_after_step  False: observe right after mj_step, one physics step behind the positions, as some
                        trainers do
    camera              camera(...): the viewer's start; None: the default
    layout              names of the geoms whose model fields (geom_pos, geom_quat, geom_size, geom_rgba) reset
                        changes, a scene with several layouts (a circuit per start state): the viewer redraws
                        them after each reset
    """

    id = None
    label = None
    agents = [agent("robot")]
    decimation = 1
    episode_s = None
    commands = []
    settings = []
    states = []
    forward_after_step = True
    camera = None
    layout = []

    def __init__(self, sim, robots):
        self.robots = robots
        self.command_values = {c["name"]: np.array(c["default"], np.float64) for c in self.commands}

    def set_command(self, name, value):
        """A command's value, from the extension's widgets or the keyboard."""
        self.command_values[name] = np.asarray(value, np.float64)

    def randomize(self, sim, rng):
        """Draw the scene's physical parameters (frictions, an object's mass): write absolute values, from the
        nominal ones kept in __init__. rng None: the nominal values."""

    def reset(self, sim, rng, state=None):
        """Set the start state: sim.data.qpos, qvel, the commands, the model fields of the scene's layout. state:
        an index into states, None to draw one. Clear the observation's memory (history, delays)."""

    def after_step(self, sim):
        """Once per policy step, after the physics: time-varying commands, the observation's memory."""

    def observe(self, sim, name, rng):
        """The observation of robot `name` (float32), with noise if sim.options['noise']."""
        raise NotImplementedError

    def done(self, sim):
        """None while the episode goes on, else done(...). By default: time_out after episode_s."""
        if self.episode_s is not None and sim.data.time >= self.episode_s - 1e-9:
            return done("time_out", "Time out")
        return None

    def metrics(self, sim):
        """metric(...)s for the extension's panel, after each policy step."""
        return []

    def markers(self, sim):
        """arrow(...)s and point(...)s to draw in the scene, after each policy step."""
        return []


__all__ = ["CONTRACT_VERSION", "OBJECT_KINDS", "Robot", "Sim", "Task", "agent", "arrow", "camera", "command", "done",
           "field_shape", "metric", "point", "setting"]
