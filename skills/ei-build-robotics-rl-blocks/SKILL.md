---
name: ei-build-robotics-rl-blocks
description: Build an Edge Impulse custom learning block that trains a robot policy with reinforcement learning in simulation (MuJoCo, Gymnasium, Stable-Baselines3 or another trainer), using the shared ei_rl package. Covers the block layout, the Studio contract for a block without training data, the environment contract, training on a wall-clock budget, the ONNX export of the actor, the training telemetry for the RL training monitor extension (rl-viz, the task descriptor, snapshots), the simulation bundle and live checkpoints for the RL policy simulator extension (ei-rl-sim), the sim-to-sim evaluation, the policy/ folder (checkpoint, configs, DEPLOY.md) and its download from Studio's Deployment tab (deploy-block-rl-policy), the Dockerfile, pinning, asset licenses, local tests and pushing. Use when asked to create, port, extend or debug an RL block for a new robot, task or trainer, or to add the monitor or simulator outputs to an existing one.
---

Help the user build an Edge Impulse custom learning block that trains a robot policy by reinforcement learning.
The block simulates the robot itself, so it learns from no project data. It writes `model.onnx` for Studio, plus
two outputs for the Studio extensions:

- the **training telemetry** (format `rl-viz`), read by the
  [RL training monitor](https://github.com/edgeimpulse/rl_training_monitor_extension): learning curves, reward
  terms, clips of the policy at training milestones, the benchmark of the exported model;
- the **simulation bundle** (format `ei-rl-sim`), read by the
  [RL policy simulator](https://github.com/edgeimpulse/rl_policy_simulator_extension): a copy of the simulation
  the policy trains in (`sim/scene.mjb.gz`, `sim/visual.bin.gz`, `sim/sim.json`, and the block's Robot and Task
  in `sim/python/`), so the extension runs the policy in the browser, plus the **checkpoints** (`checkpoints/step_<n>.onnx`) it saves while it trains.

The shared code is in this repository (`ei_rl`). A block keeps only what is specific to its robot and task.
For the general learning-block rules (parameters.json, arguments, output files, Docker, push), see the
`ei-build-custom-learning-blocks` skill; this one adds what is specific to RL.

## The formats belong to the extensions

`rl-viz` is specified in the training monitor's README (sections "Data format", "Task descriptor", "Adding a
robot"), `ei-rl-sim` in the policy simulator's README ("Bundle format", "Live checkpoints", "Parity test").
`ei_rl` implements both. Never invent a field or a file in a block: if a task needs something new (a monitor
column, a field of the Sim), it is agreed with the extension's owner, added to the extension, then to `ei_rl`. A
block emits the formats only through `ei_rl`, never with its own copy. A new robot or task in the simulator needs
none of that: it is the block's own Robot and Task on `ei_rl.sim` (see [The simulation bundle](#the-simulation-bundle)).

Read those README sections before writing a task descriptor or a simulation.

## Layers

| Layer | Holds | Example |
|---|---|---|
| `ei_rl` (this repo) | The Studio contract, the two formats, the simulation contract (`ei_rl.sim`), the trainer integrations | `RLViz`, `RLVizCallback`, `export_onnx`, `ei_rl.sim.Task`, `write_bundle` |
| The robot package | The MJCF scene, the Robot and the Task on `ei_rl.sim` (`robot.py`, `env.py`), the environments around them, the rewards, registered as tasks the framework's way; its own train and play commands. No Studio code (`ei_rl.sim` needs only numpy) | `microduck_rl` (mjlab tasks, the robot maker's); `miniauto_rl` (Gymnasium tasks, in the block's repo) |
| The block | The Studio entry point, which tasks Studio lists, the block's own tasks, and the task-specific glue for the extensions | `ei_train.py`, `studio.toml`, `tasks/`, `rl_viz.py`, `sim_bundle.py`, `Dockerfile` |

A typical block:

| File | Purpose |
|---|---|
| `<robot>_rl/` (or a pinned dependency) | The robot package (see [Tasks](#tasks)) |
| `pyproject.toml`, `LICENSE` | Make the package in the block's repo installable (`pip install -e .`) |
| `ei_train.py` | The ENTRYPOINT: arguments, the task, training, export, verification, evaluation, metrics, `policy/` |
| `studio.toml`, `studio.py` | Which tasks Studio lists, their labels, the default; `studio.load()` registers every task and returns that list |
| `tasks/` | The block's own tasks, in the robot package's format; `_template.py`, the code editor's starting code; `python -m tasks train\|play\|list-envs` |
| `reward_params.py` | Generates `parameters.json` (the Task option, the reward weights) from the tasks, `--check`, and applies the weights set in Studio |
| `rl_viz.py` | `task_info()` (the task descriptor) and `rollout_snapshot()` (the test each snapshot plays) |
| `<robot>_rl/robot.py`, `<robot>_rl/env.py` | The Robot and the Task on `ei_rl.sim`: the simulation training steps and the simulator runs |
| `sim_bundle.py` | `write_bundle(out_dir)`: the physics and visual models, `task_spec`, `python_files`; run in the Dockerfile |
| `record_samples.py` | Optional: model inputs and outputs as Studio samples |
| `parameters.json` (generated), `Dockerfile`, `requirements-block.txt` | The block |
| `requirements.txt`, `setup.sh` | A local venv with the same pins, and the package installed editable |

Start from the closest existing block and replace the robot and the task, rather than from scratch.

## Naming

- **The block** (its repo and folder): `learning-block-rl-<robot>`, the robot's name in lowercase, one word,
  e.g. `learning-block-rl-microduck`. Not the task or the trainer: one block trains all its robot's tasks, and
  a new task is a task in that block, not a new block.
- **The shared code:** `learning-block-rl-core` (this repo), Python package `ei_rl`, `ei-rl` to pip.
- **A robot package:** `<robot>_rl` (e.g. `microduck_rl`), or the name its maker publishes it under.
- **The deployment block:** `deploy-block-rl-policy`, one for every RL block.

Every RL repo starts with `learning-block-rl-`, so the blocks and their shared code sort together. Older blocks
named `custom_block_rl_*` take the new name when they move to `ei_rl`; ask the user before renaming a repo.

## The Studio contract for RL

- **No training data.** Declare `"operatesOn": "self_contained_freeform_output"`: the impulse has only the
  learn block, with no input or processing block, and the project needs no data. Say in the description that
  the block needs no project data.
- **Arguments.** Use `ei_rl.core.parser` and `parse_args` (they add `--data-directory`, `--out-directory`,
  `--custom-data-directory` and ignore unknown arguments). Every `parameters.json` entry has a matching argument
  with the same default. Values arrive as strings: give `type=` in argparse.
- **Typical parameters:** training timesteps (millions), max training time (minutes), learning rate, the task's
  own settings (episode length, a curriculum), parallel simulations (0 = automatic), seed; experts' settings under
  `"section": "advanced"`.
- **Outputs** in `--out-directory`: `model.onnx`, `custom_training_metrics.json` (the Experiments tab), and any
  file for deployment (e.g. `deploy_config.json`: the control rate, action scaling, default joint positions),
  ideally in `policy/` (see Deployment).
- **Custom data** in `/home/custom_data` (`custom_data_directory(args, out)`, which returns `None` when it can't
  be used): the telemetry, the bundle, the checkpoints. Studio keeps it with the learn block and the extensions
  read it, even while the job runs.
- **No network at runtime.** Everything the block needs is in the image.
- **Container limits.** `os.cpu_count()` is the host's, not the job's: use
  `default_workers(worker_bytes, main_bytes)`, which reads the cgroup CPU quota and memory limit. Measure a
  worker's memory (each subprocess importing torch costs about 0.45 GB).

```python
from ei_rl.core import (custom_data_directory, default_workers, describe_dataset, out_directory,
                              parse_args, parser, write_training_metrics)

p = parser("My robot, PPO (Edge Impulse custom learning block)")
p.add_argument("--timesteps-millions", type=float, default=10.0)
p.add_argument("--max-minutes", type=float, default=60.0)
p.add_argument("--n-envs", type=int, default=0)
p.add_argument("--seed", type=int, default=0)
args = parse_args(p)
out = out_directory(args)
n_envs = args.n_envs or default_workers(worker_bytes=0.6e9, main_bytes=1.0e9)
describe_dataset(args.data_directory, note="not used: experience comes from the simulation")
custom_data = custom_data_directory(args, out)
```

## The environment

The task lives in the environment; everything downstream reads it. For `RLVizCallback`, the env provides:

- `reward_scales`: reward term -> weight, as an attribute (it orders and weights the monitor's reward terms);
- `control_dt`: the policy period in seconds;
- each term's **unweighted** value in the step `info`, under the same name; with reward
  `sum(scale * term) * control_dt`, the monitor shows `scale * mean(term)`, reward per second of simulated time;
- `is_failure` / `is_success` in the `info` of an episode's last step (or pass `failure=` / `success=` functions
  to the callback; `ended_early` counts every termination that isn't a time-out as a failure).

```python
class MyRobotEnv(gym.Env):
    def __init__(self, control_dt=0.02, randomize=True, ...):
        self.control_dt = control_dt
        self.reward_scales = dict(track_velocity=1.5, upright=0.5, action_rate=-0.01, fall=-10.0)

    def step(self, action):
        ...
        terms = dict(track_velocity=..., upright=..., action_rate=..., fall=float(fell))
        reward = sum(self.reward_scales[k] * v for k, v in terms.items()) * self.control_dt
        info = dict(terms, is_failure=fell, is_success=reached)
        return obs, reward, fell, timed_out, info
```

Build its simulation on `ei_rl.sim`: the env wraps a Host stepping the block's Robot and Task, and adds the
rewards (see [The simulation bundle](#the-simulation-bundle)). Design it for sim-to-real from the start, since the
simulator runs the Robot and the Task as they are:

- an observation from what the real robot senses, with its noise and delay; a privileged critic may see more
  (the true state), appended after the actor's part, so the export can slice it off;
- the robot's sensors as **MuJoCo sensors** in the model (`gyro`, `accelerometer`, `framezaxis`, `rangefinder`,
  `geomdist`, `touch`, ...), read from `sim.data.sensordata`, rather than computed in Python: the bundle carries
  them in the `.mjb`, they cost no Python in the browser, and they survive a MuJoCo upgrade. Add
  noise, delay and thresholds on top (a line sensor: a `rangefinder` per channel, compared with a distance).
  The same goes for terminations a sensor can state (off the line: a `geomdist` over a distance). Sensors cost
  time in every `mj_step`; if the policy reads them once per control step, disable them on the model before it
  is written (`model.opt.disableflags |= mjDSBL_SENSOR`) and let the host's `mj_forward` after the substeps
  compute them;
- a scene the training draws at random (a circuit, obstacles): move the model's geoms (`geom_pos`, `geom_quat`,
  `geom_size`, `geom_rbound` for `mj_ray`, `geom_rgba`) in `Task.reset` rather than recompiling, and list them in
  the Task's `layout`, so the viewer redraws them;
- actions in [-1, 1], mapped to targets the same way in training, in the simulator and on the robot (the Robot's
  `command` and `apply`);
- the actuator model (servo, PD, the firmware's own mixing) and domain randomization (friction, masses,
  motor parameters, latency), with nominal values for the snapshots and ranges for the evaluation;
- a `randomize=False` mode and a seeded `reset(seed=...)`, so a snapshot or a test is the same every time (the
  Host's `randomize` option and its `rng`).

## Tasks

A **task** is a registered ID, the framework's way: `register_mjlab_task(...)` for mjlab, `gym.register(...)` for
Gymnasium. The block trains tasks by ID, so a task that trains with the robot package's own `train` command
trains in the block, and a new task is a registration, not new block code.

**The robot package** (`<robot>_rl`) registers its tasks on import and has its own `train` and `play` commands. It
imports nothing from Edge Impulse, so it works without Studio; the block installs it unchanged (pinned to a commit,
or from the block's repo with a `pyproject.toml`). An mjlab package declares an entry point in the `mjlab.tasks`
group, which mjlab imports. A Gymnasium package:

```python
# miniauto_rl/__init__.py
TASKS = ("MiniAuto-Football-v0",)
if "MiniAuto-Football-v0" not in gym.registry:     # registering an ID again only warns: guard it
    gym.register("MiniAuto-Football-v0", entry_point="miniauto_rl.football:MiniAutoFootballEnv")

def env_creator(task_id, **kwargs):
    """(the class that builds the task, its registration's kwargs updated with kwargs)"""
    spec = gym.spec(task_id)
    creator = load_env_creator(spec.entry_point) if isinstance(spec.entry_point, str) else spec.entry_point
    return creator, {**spec.kwargs, **kwargs}

def make(task_id, **kwargs):                       # gymnasium.make() without its wrappers
    creator, kw = env_creator(task_id, **kwargs)
    return creator(**kw)
```

- Use `make()`, not `gymnasium.make()`, where the code calls the env's own methods or its MuJoCo model (the
  evaluation, the snapshots, the bundle): the wrappers `gymnasium.make()` adds hide them.
- Give `SubprocVecEnv` the creator and its kwargs (`make_vec_env(creator, env_kwargs=kw, vec_env_cls=SubprocVecEnv)`),
  not the task ID: a worker process doesn't have the registrations of the block's own tasks or the code editor's.
- A variant (other reward weights, another episode length) is another registration that reuses the env with other
  kwargs; a new reward term is a subclass, registered under its own ID.

**Studio's Task option.** `studio.toml` is the only Edge Impulse file about tasks, and it stays in the block:

```toml
[tasks."MiniAuto-Football-v0"]
name = "Football, one on one (self-play)"   # the label in Studio's Task option
default = true                               # the task the form starts with
```

`studio.load(code_file=None)` imports the robot package and every `tasks/*.py` not starting with `_` (each one
registers its tasks), then returns `{task ID: Task}` in the order Studio lists them; a task in `tasks/` that
`studio.toml` doesn't list comes after, under its ID. `python -m tasks train|play|list-envs` runs the robot
package's commands with the block's tasks registered, so a task is tested before it is pushed.

**Reward weights in Studio.** `reward_params.py` generates `parameters.json` from the tasks: the Task option, and
for each task its reward weights, read from the environment it builds, under `"section": "advanced"` with a
`showIf` on the task (param `reward-<task slug>-<term>`, the default being the task's weight). Run it after any
change to a task or to `studio.toml`; `reward_params.py --check`, in the Dockerfile, fails when `parameters.json`
is out of date. At training, `apply(task_id, args)` returns the env kwargs for the weights that differ from their
default (so an untouched form trains the task as registered) and the list of changes, for the run info and
`DEPLOY.md`. It ignores the other tasks' fields (Studio may pass them), and rejects a weight whose sign is the
opposite of the default's.

- argparse reads `-1e-05` as an option: join each value to its flag first (`--reward-x=-1e-05`,
  `reward_params.join_values(sys.argv[1:])`).

**Your own task in Studio's code editor.** The Task option ends with "Your own task (code editor)" (value
`code-editor`), and a **Task code** parameter of type `code_editor` (`"codeEditorLanguage": "python"`, `showIf`
on it) whose default is `tasks/_template.py`: a task module, commented for a user, that subclasses the robot's
env and registers exactly one task under a new ID. It may set `STUDIO = dict(name=...)` in place of a
`studio.toml` entry. `ei_train.py` saves the code as `policy/task.py` and calls `studio.load(that_file)`, which
reports a syntax error with its line and fails unless the code registers exactly one task. Its reward weights are
in its code: no weight fields show for it.

- With `SubprocVecEnv`, the class defined in that file must reach the workers: `cloudpickle.register_pickle_by_value(module)`
  after importing it (the module isn't importable in the workers).
- Check the template in the Dockerfile: it must register a task that builds and resets.

## Training

**Stable-Baselines3:** add `RLVizCallback` to `learn`. It logs every rollout, plays the snapshots, schedules the
checkpoints and finishes the run.

```python
from ei_rl.sb3 import RLVizCallback
from ei_rl.viz import RLViz

viz = RLViz(custom_data, total_timesteps=args.timesteps_millions * 1e6, max_minutes=args.max_minutes,
            task=task_info(), checkpoint=checkpoint,              # see the bundle section
            run_info=dict(task=task_id, n_envs=n_envs, seed=args.seed))
callback = RLVizCallback(viz, rollout=lambda policy, render: rollout_snapshot(task_id, policy, render=render))
model.learn(total_timesteps=int(args.timesteps_millions * 1e6), callback=[callback, WallClockLimit(args.max_minutes)])
```

`rollout(policy, render)` gets `policy(obs) -> action` on raw observations (the callback applies the current
`VecNormalize` statistics) and returns `(frames, stats)`.

**mjlab + rsl_rl:** `ei_rl.mjlab.RslRlViz` wraps the runner's logger (rsl_rl has no callbacks). It reads the
reward weights and the time-out terminations from the task's managers, logs every PPO iteration, and stops at
the time budget.

```python
from ei_rl.mjlab import RslRlViz, checkpoint

viz = RLViz(custom_data, total_timesteps=iterations * num_steps_per_env * num_envs, max_minutes=args.max_minutes,
            task=task_info(), checkpoint=checkpoint(runner, custom_data))
hook = RslRlViz(viz, rollout=rollout_snapshot)   # rollout_snapshot(policy, render) -> (frames, stats)
hook.attach(runner, env)
model_pt = hook.learn(iterations, init_at_random_ep_len=True)
hook.finish()
```

The rollout gets the runner's deterministic policy (in eval mode) and plays it in the block's own test env.

**Another trainer** (skrl, a custom loop): call `RLViz` yourself and set its `rollout`.

```python
viz.rollout = lambda render: rollout_snapshot(current_policy(), render=render)
viz.start(control_dt=env.control_dt, reward_weights=env.reward_scales, success=True)
for it in range(iterations):
    ...  # collect, update
    viz.log(step, dict(ep_rew=..., ep_len=..., fps=..., fall_rate=..., success_rate=..., entropy=..., lr=...),
            terms={k: weighted_mean_per_second[k] for k in env.reward_scales})
    if viz.out_of_time():
        break
viz.finish(step)
```

**The wall-clock budget.** Studio jobs have a time limit. Stop at `--max-minutes` whatever the timesteps, then
export what you have. `RLVizCallback` doesn't stop training: add a callback whose `_on_step` returns `False` past
the deadline (`WallClockLimit` in the miniAuto block's `miniauto_rl/train.py`), or check `viz.out_of_time()` in your loop. Snapshots follow whichever of timesteps and time is further.
Say in the parameter's help to keep it below the job limit, and in the README what a run gives at a few budgets.

## Exporting the policy

The ONNX contract the extensions and the deployment rely on: input `"input"` `[1, obs_dim]` float32, output
`"output"` `[1, act_dim]` in [-1, 1], observation normalization baked in, the actor only, opset 10, a single file.

```python
from ei_rl.sb3 import export_onnx, verify_onnx

export_onnx(model, vec_norm, out / "model.onnx", obs_dim=OBS_DIM)    # obs_dim: the actor's part, with a privileged critic
verify_onnx(model, vec_norm, out / "model.onnx", make(task_id, randomize=False), obs_dim=OBS_DIM)  # vs model.predict, 1e-4
```

Another trainer writes its own exporter to the same contract and checks it against its own inference the same
way. Always verify: a wrong normalization or slice gives a model that loads and acts badly.

## The training monitor

The block gives `RLViz` two task-specific things, usually in `rl_viz.py`.

**`task_info()`**, the task descriptor: how the monitor names the task and what it shows. Every field is
optional; see "Task descriptor" in the monitor's README.

```python
def task_info(episode_seconds=5.0):
    return dict(
        name="SO-101 push cube to target",
        summary="Push a 35 mm cube 6–12 cm onto a target within 2 cm",
        episode=f"one episode is {episode_seconds:g} s of pushing",
        success=dict(label="Success rate"),        # what success_rate counts
        failure=dict(label="Cube lost"),           # what fall_rate and a snapshot's fell_at mean here
        snapshot=dict(
            help="The deterministic policy, nominal physics, the same layout every time.",
            stats=[dict(key="final_distance", label="Final distance", unit="cm", scale=100, digits=1),
                   dict(key="success_at", label="On target at", unit="s", digits=1)],
            trace=dict(key="cube_distance", label="Cube to target", unit="cm", scale=100, digits=1,
                       reference_key="success_radius", reference_label="Success radius")),
        eval=dict(name_label="Layout",
                  columns=[dict(key="successes", label="On target", of="episodes", good="high"),
                           dict(key="final_distance", label="Final distance", unit="cm", scale=100, digits=1)]),
    )
```

**`rollout_snapshot(policy, render)`**: the same deterministic test at every snapshot (nominal physics, a fixed
seed, a fixed command or layout), so the clips compare like with like. It returns the frames (a list of RGB
arrays) and the stats: `fps` when there are frames, `outcome` (a few words: "On target after 1.2 s"),
`outcome_kind` (`success`, `failure`, `neutral`), `fell_at` (seconds or `None`), `sim_return`, the trace (a list
per frame) and the descriptor's stat keys.

- Render with `mujoco.Renderer` at about 10 fps and 240 x 320; GIFs stay small.
- Rendering needs OpenGL: `MUJOCO_GL=osmesa` and `libosmesa6` in the image. If it fails, `RLViz` keeps the stats
  and skips the GIF; never let a snapshot stop training.
- Snapshots are at fixed fractions of the run plus "Final"; a `--no-snapshots` flag turns them off.

After training, `write_eval(custom_data, results, seconds, seeds, caption)` writes the benchmark and marks the
run finished.

## The simulation bundle

The simulator runs the block's own simulation code, not a description of it. The block writes it as two
classes on `ei_rl.sim`, the simulation contract, in two files:

- `robot.py`, a **Robot**: how one robot's actions drive its motors (ctrl, or applied forces), every physics
  step;
- `env.py`, a **Task**: the start states, what each robot observes, when an episode ends, and what the
  extension shows (commands, settings, metrics, markers).

Training steps them with `ei_rl.sim.native.Host` on mujoco; the extension steps the same files with the same
loop in Pyodide, on MuJoCo WASM. So they import only the standard library, numpy, `ei_rl.sim` and each other
(or the block's other modules that obey the same rule): no mujoco, gymnasium or torch. Keep them in modules of
their own, not in a package `__init__.py`, which the bundle leaves empty. `ei_rl.sim`'s docstring is the
specification; `tests/toy_sim/` in this repo is the smallest complete example (a cart on a rail).

**The Sim** they see: `sim.model` and `sim.data`, the MuJoCo fields in `MODEL_FIELDS` and `DATA_FIELDS` as numpy
arrays (shaped as the mujoco bindings shape them), the sizes in `MODEL_SIZES`, `sim.data.time`; `physics_dt`,
`decimation`, `dt`; `options`; and `name2id`, `id2name`, `id` (a missing name is a `KeyError`), `ray` (mj_ray)
and `forward` (mj_forward). The data they may write is `WRITABLE_DATA` (`qpos`, `qvel`, `act`, `ctrl`,
`qfrc_applied`, `xfrc_applied`, `mocap_pos`, `mocap_quat`), plus any model field. A field outside the contract
is an `AttributeError`; one more is a change to `ei_rl.sim` and to the extension (`CONTRACT_VERSION`).

**The order.** `Host.reset(state)` returns `{agent: observation}`, `Host.step(actions)` that and `done`:

```text
reset(state)    mj_resetData
                robot.randomize(sim, rng) for each robot, then task.randomize(sim, rng)   rng None: randomization off
                task.reset(sim, rng, state)
                robot.reset(sim, rng) for each robot
                mj_forward
                task.observe(sim, name, rng) for each robot             -> {name: obs}

step(actions)   robot.command(sim, action) for each robot               the action of action_delay steps ago
                decimation times: robot.apply(sim) for each robot, then mj_step
                mj_forward                                              unless task.forward_after_step is False
                task.after_step(sim)
                task.observe(sim, name, rng) for each robot             given obs_delay steps later
                task.done(sim)                                          -> ({name: obs}, done or None)
```

`rng` is a numpy Generator the host seeds: the same seed gives the same random draws in training and in the
browser (and the same episode when their MuJoCo versions match), so draw every random number from it. `sim.options` holds the host's switches (`noise`, `randomize`, `obs_delay`,
`action_delay`) and the declared settings' values. The two delays are the host's: whole policy steps on top of any
latency the robot models itself, which stays the block's (per sensor, randomized as trained).

**The declarations** are plain dicts from `ei_rl.sim`'s functions, set as class attributes:

| On | Attribute | What |
|---|---|---|
| Task | `id`, `label` | the task's ID (as in the task descriptor) and a readable name |
| Task | `agents = [agent(name, label=None, drivers=("policy", "keyboard", "none"))]` | the robots; the first is the policy's, each other one is driven by one of its `drivers`, picked in the extension. Default: one agent, `robot` |
| Task | `decimation`, `episode_s` | physics steps per policy step; the episode length (s), for the default `done()` (None: no limit) |
| Task | `commands = [command(name, dim, low, high, default=None, label=None, axes=None, keys=None, widget="slider")]` | what the user sets while the policy runs and the policy observes; `widget` `slider` (one per axis), `buttons` (`low`, `high`, `default` as presets) or `point` (x, y picked on the ground); `keys` `{key: [axis, value]}` while held. The current values are in `self.command_values[name]` |
| Task, Robot | `settings = [setting(name, low, high, default, label=None, unit=None, step=None)]` | a value the user sets (a payload, a motor lag), read from `sim.options[name]` at the next reset; the policy doesn't observe it |
| Task | `states` | labels of the start states the extension offers; `reset`'s `state` is an index into it, None to draw one |
| Task | `forward_after_step` | False: observe right after `mj_step`, as some trainers do |
| Task | `camera = camera(distance=1.0, elevation_deg=-20.0, azimuth_deg=135.0, lookat=(0, 0, 0.1), track=None)` | the viewer's start; `track`: a body name to follow |
| Task | `layout` | the names of the geoms whose `geom_pos`, `geom_quat`, `geom_size` or `geom_rgba` a reset changes (a circuit per start state); the viewer redraws them after each reset |
| Robot | `action_dim`, `keys` | the action's length; `{key: [action index, value]}`, how the keyboard drives it (held keys add up) |

And what they return at run time:
- `Task.done(sim)`: None, or `done(reason, label=None, tone="info", **info)`, `tone` `success`, `failure` or
  `info`. The default returns `done("time_out", "Time out")` after `episode_s`; call `super().done(sim)` last.
  The trainer maps `time_out` to truncation and any other reason to termination;
- `Task.metrics(sim)`, `Robot.metrics(sim)`: `metric(label, value, unit=None, chart=True, low=None)`s, the
  extension's tiles and chart;
- `Task.markers(sim)`: `arrow(pos, vec, rgba=..., label=None)`s and `point(pos, rgba=..., radius=0.02, label=None)`s,
  drawn in the scene.

The methods: `Robot(sim, name)` and `Task(sim, robots)` (the host makes the robots first; `self.robots` is
`{agent name: Robot}`, so the task reads its robots there) look up their ids in `__init__` (`sim.id("joint", "slide")`),
never hardcoded (the bundle's model has fewer geoms than the full one). `randomize` writes absolute values from
the nominal ones kept in `__init__` (`mj_resetData` doesn't reset the model), and the nominal values when
`rng` is None. `Task.reset` sets `qpos`, `qvel`, the commands and the layout, and clears the observation's
memory; `Robot.reset` clears the actuator's (filters, delays). `Task.observe(sim, name, rng)` returns float32,
with noise only if `sim.options["noise"]`. Anything stateful the policy sees (history, delays, a filtered
command) lives in these classes, or the browser won't have it.

```python
# myrobot_rl/robot.py
import numpy as np
from ei_rl.sim import Robot, metric, setting

class MyRobot(Robot):
    action_dim = 12
    keys = {"w": [0, 1.0], "s": [0, -1.0]}
    settings = [setting("payload_kg", 0.0, 2.0, 0.0, "Payload", "kg")]

    def __init__(self, sim, name):
        super().__init__(sim, name)
        self.acts = [sim.id("actuator", n) for n in ACTUATORS]
        self.base = sim.id("body", "base")
        self.mass0 = float(sim.model.body_mass[self.base])

    def randomize(self, sim, rng):
        sim.model.body_mass[self.base] = self.mass0 + sim.options["payload_kg"]

    def reset(self, sim, rng):
        self.target = np.zeros(self.action_dim)

    def command(self, sim, action):
        self.target = OFFSET + SCALE * np.clip(action, -1, 1)

    def apply(self, sim):
        sim.data.ctrl[self.acts] = self.target
```

```python
# myrobot_rl/env.py
import numpy as np
from ei_rl.sim import Task, camera, command, done, metric

class Walk(Task):
    id, label = "MyRobot-Walk-v0", "Walk"
    decimation, episode_s = 4, 20.0
    commands = [command("twist", 3, [-0.5, -0.3, -1.0], [1.0, 0.3, 1.0], label="Velocity",
                        axes=["forward", "left", "turn"], keys={"ArrowUp": [0, 0.5]})]
    camera = camera(1.5, -20, 135, track="base")

    def reset(self, sim, rng, state=None):
        sim.data.qpos[:] = sim.model.key_qpos[0]
        self.last = np.zeros(12)

    def observe(self, sim, name, rng):
        x = np.concatenate([...])
        if sim.options["noise"]:
            x = x + rng.uniform(-NOISE, NOISE, x.shape)
        return x.astype(np.float32)

    def done(self, sim):
        if tilt(sim) > TILT_LIMIT:
            return done("fell", "Fell", "failure")
        return super().done(sim)

    def metrics(self, sim):
        return [metric("Forward speed", forward_speed(sim), "m/s")]
```

**Training on it.** The block's Gymnasium env wraps a native Host, and keeps what the browser doesn't need:
the rewards, the reward weights, the info for the monitor. `host.model` and `host.data` are mujoco's own
`MjModel` and `MjData`, for the rewards and for rendering; the Robot and the Task see only the contract.

```python
from ei_rl.sim.native import Host

class MyRobotEnv(gym.Env):
    def __init__(self, randomize=True, seed=0):
        self.host = Host(build_model(), Walk, MyRobot, dict(randomize=randomize), seed)
        self.control_dt = self.host.sim.dt
        self.reward_scales = dict(track_velocity=1.5, upright=0.5, action_rate=-0.01, fall=-10.0)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.host.rng = np.random.default_rng(seed)
        return self.host.reset()["robot"], {}

    def step(self, action):
        obs, end = self.host.step(action)          # one robot: its action alone; several: {name: action}
        terms = self.reward_terms(self.host.data, self.host.task)
        reward = sum(self.reward_scales[k] * v for k, v in terms.items()) * self.control_dt
        failed = end is not None and end["reason"] != "time_out"
        info = dict(terms, is_failure=failed)
        return obs["robot"], reward, failed, end is not None and not failed, info
```

`Host(model, task_cls, robot_cls, options=None, seed=0)`: `options` overrides `DEFAULT_OPTIONS`
(`noise=True, randomize=True, obs_delay=0, action_delay=0`) and the settings' defaults. `host.task`,
`host.robots` (`{name: Robot}`), `host.sim`, `host.metrics()` and `host.markers()` are there for the snapshots
and the evaluation. A trainer with its own vectorized simulator (mjlab, on GPU) can't step a Host: its Robot
and Task then reproduce the training env, and the block checks them against it.

**Writing the bundle** (`sim_bundle.py`, called by `ei_train.py` and the Dockerfile):

```python
import mujoco
from ei_rl import bundle
from myrobot_rl.env import Walk
from myrobot_rl.robot import MyRobot

def write_bundle(out_dir):
    spec = mujoco.MjSpec.from_file(SCENE_XML)
    full = spec.compile()
    vis_bin, visual = bundle.visual_model(full)
    model, mjb = bundle.physics_model(full, spec)
    return bundle.write_bundle(out_dir, bundle.task_spec(Walk, MyRobot, model), mjb, vis_bin, visual,
                               bundle.python_files(Walk, MyRobot))
```

1. **Physics:** `bundle.physics_model(full, spec, reset=None, control=None, ...)` compiles the scene without its
   visual-only geoms, checks it simulates exactly as the full model under random controls, and returns
   `(model, mjb bytes)`. It drops every geom with `contype = conaffinity = 0`, including those a sensor or a ray
   sees: when the task needs them, write the model it trains on with `bundle.model_bytes(model)` instead.
2. **Visuals:** `bundle.visual_model(full)` decimates the meshes into `visual.bin` and lists every visual geom,
   by name (the layout's geoms are matched by name). A mesh that may not be redistributed goes in
   `visual["remote_meshes"]` instead (see Assets; `with_group=True` gives each geom its group).
3. **sim.json:** `bundle.task_spec(task_cls, robot_cls, model)` reads it from the classes: the entry points,
   the timing, the agents, commands, settings, keys, states, camera and layout, and the observation and action
   sizes. It runs one reset and one step (no noise, no randomization) to check them, and fails if a layout
   geom or the tracked body isn't in the model.
4. **python/:** `bundle.python_files(task_cls, robot_cls)` collects their modules, the block's modules they
   import, and `ei_rl.sim` (without `native.py`). It raises `ValueError`, listing every import Pyodide can't
   run.
5. **`bundle.write_bundle(out_dir, spec, mjb, vis_bin, visual, python)`** adds `python.files` and the
   `bundle_id`, and swaps `<out_dir>/sim/` in whole. It returns the sim directory.

**The parity check.** The training code is the bundle's code, so what's left to check is the code on the
browser's MuJoCo. In the simulator's checkout, `test/reference.py <sim dir> ref.json` records a native rollout
of the bundle (`ei_rl.sim.native.from_bundle`), and `npm run parity -- <sim dir> ref.json` replays it in the
browser's engine; observations and qpos must match within 1e-9. Run `from_bundle` in a fresh process: it
imports the bundle's modules by name, and a process that has already imported the block's gets those instead.
In the Dockerfile, writing the bundle is the check that fails the build: `task_spec` resets and steps the
task, and `python_files` refuses an import the browser lacks.

**Live checkpoints.** In `ei_train.py`, before training: `clear_policies(custom_data)`, write the bundle, then
`print_live_header(sim_dir)`. RLViz calls the checkpoint function with the snapshots, at least every 15 minutes,
and at the end; after training, `write_policy(custom_data, onnx_path)`.

```python
from ei_rl.bundle import clear_policies, print_live_header, save_checkpoint, write_policy

live = False
if custom_data is not None:
    clear_policies(custom_data)
    try:                                   # the bundle is optional: never stop a training over it
        print_live_header(sim_bundle.write_bundle(custom_data))
        live = True
    except Exception as err:
        print(f"WARNING: no simulation bundle for the policy simulator extension: {err!r}")

def checkpoint(step, iteration, label):
    save_checkpoint(custom_data or out, step, iteration, label,
                    lambda path: export_onnx(callback.model, callback.training_env, path, obs_dim=OBS_DIM), live)
```

**One MuJoCo version.** A `.mjb` only loads in the MuJoCo version that compiled it, and the extension ships each
version a block pins. Moving a block to a new MuJoCo means adding that version to the extension first.

**MuJoCo 4.** The extension already ships 4.0.1. To move a block: pin `mujoco==4.0.1` in a test venv, run
`sim_bundle.py` and the simulator's parity test, then change `requirements.txt` and retrain (a policy
trained on 3.x usually still plays, but train on what you ship). Keep code that also runs on 3.x until then:
build the model with `MjSpec`, use enums by name (`mujoco.mjtSensor.mjSENS_RANGEFINDER`), and set what recent
versions require explicitly (a `rangefinder` made with `MjSpec` needs `intprm[0] = 1 << mjRAYDATA_DIST`, the
data it outputs, in 3.15 and 4.0 alike). A block on mjlab or another framework moves when that framework supports MuJoCo 4.

## Evaluation and metrics

After the export, run the **ONNX model**, not the trainer's policy, on a fixed benchmark: randomization and noise
on, fixed seeds, the same whatever the block's parameters, so impulses compare. Several conditions (opponents,
commands, layouts) make the rows of `eval.json`; each result has a `name` and the descriptor's eval columns.
Put a few headline numbers in `custom_training_metrics.json` with `write_training_metrics(out, {...})` (success
rate, failure rate, timesteps trained, training time), which Studio shows in the Experiments tab.

**Studio samples** (optional): `write_samples` and `write_io_pairs` turn the benchmark's model inputs and outputs
into acquisition files and an `info.labels`, so the project has samples to test a deployment against.

## Deployment

What the robot needs besides `model.onnx` goes in a `policy/` folder, which the block writes to the out
directory and copies to the custom data (`custom_data/policy/`, where the simulator also reads `model.onnx`):

- `model.onnx`, and the trainer's checkpoint it was exported from (`model_<n>.pt`, `model.zip`), to resume,
  fine-tune or re-export it. Only that one: the trainer's other checkpoints stay in the job's work directory,
  which is gone after the job;
- the configs it trained with: rsl_rl's `params/env.yaml` and `params/agent.yaml`; for Stable-Baselines3,
  `params/env.json` (the task ID and the env's kwargs, with the reward weights) and `params/agent.json` (the PPO
  settings), and `vecnormalize.pkl`, the normalizer the checkpoint needs;
- the robot runtime's settings, if the block has them (`deploy_config.json`: the control rate, the input layout,
  the action scaling);
- `DEPLOY.md`, how to use this policy (below);
- the task's code when the user wrote it in Studio's code editor (`task.py`).

### DEPLOY.md

Every block writes one. It is what someone who downloads the policy reads first, often without the block's
README, so it says everything needed to go from the files to a robot. `ei_train.py` generates it at the end of
the run, never as a static file: the task ID, the checkpoint's name, the settings and the numbers are the run's,
so every command can be copied as it is.

| Section | What it says |
|---|---|
| `# <task ID>` | One line: which block trained it, and with what (the robot package's recipe, the block's own task, the code editor's `task.py`) |
| Files | Each file in `policy/`. For `model.onnx`: the observation and action (dimensions, what the output means, e.g. joint targets as `offset + scale * action`), normalization baked in, how it was verified. For the checkpoint: how to resume, fine-tune or re-export it |
| Run it in simulation | The commands to play the policy in the robot package's own tools, with this run's task ID and checkpoint |
| Put it on the robot | The commands to copy, publish or load it on the robot, and what the robot's runtime needs that the model doesn't carry (the control rate, action scaling, default pose, or where `deploy_config.json` says it). For an episodic task: how to start it and what it hands back to |
| Settings changed | The Studio parameters that differ from the defaults (e.g. the reward weights: the default and the value) |
| Evaluation | The benchmark of the exported ONNX model, the numbers in `custom_training_metrics.json` |

- Read every value from the run's own objects (the task's config, the checkpoint path, the evaluation results),
  never retype it.
- A placeholder only for what the block can't know, in angle brackets (`<user>/microduck-walk`), and say what
  goes there.
- Name the robot package's version (the commit the Dockerfile pins): the commands are that version's.
- Keep it short and in plain Markdown; link to the robot package's README for the details rather than
  repeating them.

```python
def write_deploy_md(policy_dir, task_id, checkpoint, trained_with, changed, metrics):
    changes = "".join(f"- `{c['term']}`: {c['weight']:g} (default {c['default']:g})\n" for c in changed)
    (policy_dir / "DEPLOY.md").write_text(f"""# {task_id}

Trained by the Edge Impulse <robot> block with {trained_with}.

- `model.onnx`: {OBS_DIM} observations -> {ACT_DIM} joint targets, normalizer baked in.
- `{checkpoint.name}`: the checkpoint `model.onnx` was exported from.
- `params/`: the configs it trained with.

## Run it in simulation

    <the robot package's play command> {task_id} --checkpoint {checkpoint.name}

## Put it on the robot

    <the robot's own commands, with model.onnx>

""" + (f"## Settings changed\n\n{changes}\n" if changes else "") + "## Evaluation\n\n"
        + "".join(f"- {k}: {v:.4g}\n" if isinstance(v, float) else f"- {k}: {v}\n" for k, v in metrics.items()))
```

Two complete ones, both `write_policy_dir()` in the block's `ei_train.py`: the microduck block's (mjlab), with
the commands for each kind of task (perpetual or episodic); the miniAuto block's (Stable-Baselines3), with
`model.zip`, `vecnormalize.pkl` and `deploy_config.json`. Both write the commands for the robot package's tasks
(`python -m <robot>_rl.play <task ID> ...`), the block's own (`python -m tasks play ...`) and the code editor's
(copy `task.py` to `tasks/` first).

**Which deployment to pick.** For the ONNX policy only, the **Custom block** deployment type on Studio's
**Deployment** tab is enough: its zip has `model.onnx` (and its conversions). For everything about the policy
(the checkpoint, the configs with the reward weights changed in Studio, `DEPLOY.md`, the task's code with its
custom reward functions when it was written in the code editor), the user needs the deployment block [deploy-block-rl-policy](https://github.com/edgeimpulse/deploy-block-rl-policy). Say so in the block's README.

Studio's **Deployment** tab gives a deployment block only the model and its conversions, never the custom data.
The shared deployment block has `mountLearnBlock` on: Studio mounts the learn block's
folder at `/data`, and it zips the **Custom block** deployment's files with `/data/custom_data/policy/`. It
works for any RL block that writes `custom_data/policy/`, so a block doesn't need a deployment block of its own.
It needs an Enterprise organization and an impulse with a single learn block. Keep `policy/` small: the
deployment zips it as it is.

## Dockerfile

```dockerfile
FROM public.ecr.aws/g7a8t7v6/ei-custom-ml-block-base:v1.95.10
WORKDIR /app                                   # never /home or /data

# Software OpenGL for the snapshot GIFs (no GPU, no display)
RUN apt-get update && apt-get install -y --no-install-recommends libosmesa6 && rm -rf /var/lib/apt/lists/*
ENV MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa

COPY requirements-block.txt ./
RUN pip3 install --break-system-packages -r requirements-block.txt

# The robot package, its dependencies already installed above (or pinned to a commit in requirements-block.txt)
COPY pyproject.toml LICENSE ./
COPY myrobot_rl/ ./myrobot_rl/
RUN pip3 install --break-system-packages --no-deps --no-build-isolation .

COPY ei_train.py studio.py studio.toml reward_params.py parameters.json rl_viz.py sim_bundle.py ./
COPY tasks/ ./tasks/

# Fail the build early: every task builds, parameters.json is up to date, the code editor's template registers a
# task, offscreen rendering works, the bundle writes (its task resets and steps, its Python runs in the browser)
RUN python3 -c "import studio, myrobot_rl; [myrobot_rl.make(t).reset(seed=0) for t in studio.load()]"
RUN python3 reward_params.py --check
RUN python3 -c "import studio, myrobot_rl; t = [t for t in studio.load('tasks/_template.py').values() if t.from_code_editor][0]; myrobot_rl.make(t.task_id).reset(seed=0)"
RUN python3 -c "from rl_viz import rollout_snapshot; rollout_snapshot('MyRobot-Walk-v0', lambda o: [0.0] * ACT_DIM, seconds=0.2)"
RUN python3 sim_bundle.py --out /tmp/sim-check && rm -rf /tmp/sim-check

ENTRYPOINT ["python3", "-u", "ei_train.py"]
```

## Dependencies

- Pin exact versions in `requirements-block.txt` (torch CPU wheels from `--extra-index-url
  https://download.pytorch.org/whl/cpu`, mujoco, gymnasium, stable-baselines3, numpy, onnx, onnxruntime, pillow,
  pyopengl), and the same ones in the local `requirements.txt`.
- Install `ei_rl` from a fixed commit, no git needed in the image:

  ```text
  ei-rl[mujoco,sb3] @ https://github.com/edgeimpulse/learning-block-rl-core/archive/<commit>.tar.gz
  ```

  To change it alongside a block, install a checkout over the pin: `pip install -e ../learning-block-rl-core`.
- An external robot package or task library (e.g. an mjlab task) is pinned to a commit and installed at build
  time.

## Assets and licenses

- Check the license of every model, mesh and dependency before it goes in the repo, the image or the bundle.
  The bundle is a distribution: whatever is in `scene.mjb.gz` and `visual.bin.gz` is shipped to every viewer.
- A mesh that may be downloaded from where it is published but not redistributed: train on primitive collision
  shapes, keep the mesh out of the repo and the bundle, and list it in `remote_meshes` so the viewer downloads it
  from its source. `visual_model` should raise if a mesh geom slips in.
- Credit the robot's and the scene's authors in the README.

## Testing locally

- **A short run** in the image, without network, as Studio runs it:

  ```bash
  docker build -t myrobot-block .
  mkdir -p data out
  docker run --rm --network=none -v $PWD/data:/data -v $PWD/out:/out myrobot-block \
      --data-directory /data --out-directory /out --custom-data-directory /out/custom_data \
      --timesteps-millions 0.2 --max-minutes 5
  ```

  Then check `out/model.onnx`, `out/custom_training_metrics.json`, and in `out/custom_data/`: `manifest.json`
  (status `finished`), `metrics.json`, `eval.json`, `snapshots/*.gif`, `sim/`, `checkpoints/`, `policy/` (`model.onnx`, the checkpoint, `DEPLOY.md`).
- **The monitor:** in its checkout, `DEMO_DIR=/path/to/out node server.js`, then open
  `http://localhost:8080/?demo=1`. It answers the Studio API from `out/custom_data/` and the log `out.log` (save
  the run's stdout there). See its "Run locally".
- **The simulator:** in its checkout, `DEMO_DIR=/path/to/out node server.cjs` and `?demo=1`; it needs
  `custom_data/sim/` and `custom_data/policy/model.onnx`. Its parity test replays a native rollout of the bundle
  in the browser's engine (`test/reference.py`, then `npm run parity`; see "Run locally" and "Parity test").
- **Unit-sized checks** while developing: `python sim_bundle.py --out /tmp/sim`, a few episodes of the native
  Host, and `python -m unittest discover tests` in this repo.

**Refactoring a block** (moving code into `ei_rl`, renaming, reordering): run the old and the new code with
the same seed and a short training on one process, and compare everything except wall-clock fields (`time`,
`fps`, `updated`): every `[rl-...]` log line, `manifest.json`, `metrics.json`, `eval.json`, the GIFs, the
checkpoints, `model.onnx` and `sim/` byte for byte. Run the old code twice first to confirm it is deterministic.

## Pitfalls

- `parse_args` with unknown arguments: Studio passes more than the block declares.
- Worker count from `os.cpu_count()`, or one worker per CPU without checking memory: the job is killed.
- Files the extensions read while the job runs are written to a temporary name, then renamed (`ei_rl`
  does this); a block's own files there do the same.
- Leftovers of a previous run in the custom data: clear `snapshots/`, `checkpoints/` and `policy/` at the start
  (`RLViz.start` and `clear_policies` do).
- Simulation code outside the Robot and the Task (an observation filter in a wrapper, an action clip in the env):
  the simulator doesn't run it, so it runs a different robot than the one trained. Everything between the policy
  and MuJoCo goes in `robot.py` and `env.py`.
- Random numbers not from the host's `rng` (`np.random`, a second Generator): the browser's episodes no longer
  match training's for the same seed.
- Hardcoded ids instead of `sim.id(kind, name)`: the bundle's physics model has fewer geoms than the full one.
- Evaluating the trainer's policy instead of the exported ONNX: the benchmark then misses an export bug.
- A snapshot or checkpoint failure that raises: catch it, warn, and keep training.
- Stale descriptions: when the observation or action changes, update `parameters.json`, the README and
  `deploy_config.json` along with it.
- A hand-edited `parameters.json`: it is generated; edit the tasks or `studio.toml`, then run `reward_params.py`.
- Task IDs passed to `SubprocVecEnv` workers, or a code editor class pickled by reference: the workers fail to
  build the env (see [Tasks](#tasks)).

## Push

```bash
edge-impulse-blocks init    # first time only: pick the organization, writes .ei-block-config
edge-impulse-blocks push    # Edge Impulse builds the image, including the checks
```

In a project: add the block under **Create impulse** (it is the impulse's only block), then train. Set the job's CPU, memory and time limits on the block in Studio after
the push; they are not in `parameters.json`. For `model.onnx` alone, pick the **Custom block**
deployment type under **Deployment**. For the whole `policy/`, push [deploy-block-rl-policy](https://github.com/edgeimpulse/deploy-block-rl-policy) to the
organization once and pick it there (see Deployment).

## Instructions

1. Ask what the robot is (its model, actuators, sensors), the task and how success and failure are defined, the
   trainer (default: Gymnasium + Stable-Baselines3 PPO on CPU), and whether a real robot will run the policy.
   Look for an existing block with a similar robot or task to start from. Name the new block
   `learning-block-rl-<robot>` (see Naming); for a new task on a robot that has a block, add it to that block.
2. Build the robot package first, with no Edge Impulse code: the environment, its tasks registered, `train` and
   `play` (`python -m <robot>_rl.train <task ID>`, a short run that learns something). Then the block around it:
   `studio.toml`, `tasks/_template.py`, `reward_params.py` (generate `parameters.json`), and `ei_train.py`.
3. Use `ei_rl` for everything it covers; don't copy its code into the block. If the block needs a change
   in it, make it there, keeping the formats unchanged.
4. Write `task_info()` and `rollout_snapshot()` from the monitor's README, and the Robot and the Task on
   `ei_rl.sim` (start from `tests/toy_sim/` in this repo). If the task needs a MuJoCo field or function the Sim
   doesn't give, say so: it is a change to `ei_rl.sim` and the extension first.
5. Make `sim_bundle.py` write the bundle, run it in the Dockerfile, and pass the simulator's parity test.
6. Verify the ONNX export, evaluate the ONNX model on a fixed benchmark, and write the headline metrics.
7. Pin every dependency, `ei_rl` to a commit, and check asset licenses before anything is committed.
8. Write `policy/` with a `DEPLOY.md` generated for the run (see Deployment), and check it reads right for a
   task of each kind the block trains.
9. Run a short training in the image with `--network=none`, open the outputs in both extensions locally, then
   push.
10. Document in the block's README: the task, the observation and action, the rewards, the randomization, what a
    run gives at a few budgets, and the extensions' outputs.
