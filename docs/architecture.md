# RL blocks: how they work

Scope: the Edge Impulse custom learning blocks that train robot policies in simulation, the shared code they use, the two Studio extensions that read their output (training monitor, policy simulator), and the deployment block that downloads the trained policy

## Overview

An RL block trains a robot policy by reinforcement learning in a MuJoCo simulation, as an Edge Impulse custom learning block. It needs no project data: all experience comes from the simulator. Four pieces work together:

| Piece | What it is | Example |
|---|---|---|
| **Robot package** | The robot and its tasks, as standard framework code. Works without Edge Impulse | Pollen Robotics' [microduck_rl](https://github.com/pollen-robotics/microduck_rl) (mjlab) |
| **Shared code** | [`learning_block_rl_core`](https://github.com/edgeimpulse/learning_block_rl_core), Python package `ei_rl`: everything specific to Edge Impulse | `ei_rl.core`, `ei_rl.viz`, `ei_rl.bundle`, `ei_rl.mjlab`, `ei_rl.sb3` |
| **Learn block** | `learning_block_rl_<robot>`: what is pushed to Studio. A Dockerfile that pins the two above, a generated `parameters.json`, the list of tasks Studio shows, and optional tasks of its own | [learning_block_rl_microduck](https://github.com/edgeimpulse/learning_block_rl_microduck) |
| **Studio side** | Two extensions that read the block's custom data, and a deployment block that downloads the policy | [RL training monitor](https://github.com/edgeimpulse/rl_training_monitor_extension), [RL policy simulator](https://github.com/edgeimpulse/rl_policy_simulator_extension), [deploy-block-rl-policy](https://github.com/edgeimpulse/deploy-block-rl-policy) |

```
Studio ──starts──> learn block (Docker image)
                     ├─ registers the tasks: the robot package's (mjlab entry points) and the block's own (tasks/)
                     ├─ trains the selected task, through ei_rl's trainer integration
                     ├─ out directory ──> model.onnx, metrics            ──> Studio (impulse, Experiments tab)
                     └─ custom data   ──> telemetry, snapshots           ──> RL training monitor
                                          sim/, checkpoints/              ──> RL policy simulator
                                          policy/                         ──> deploy-block-rl-policy
```

The rule users rely on: **if a task trains with the framework's own `train` command, it trains in the block.**

## Naming

| What | Name | Example |
|---|---|---|
| A learn block (repo and folder) | `learning_block_rl_<robot>`: the robot's name, in lowercase, one word. Not the task or the trainer: one block trains all its robot's tasks | `learning_block_rl_microduck` |
| The shared code | `learning_block_rl_core`, Python package `ei_rl` (`ei-rl` to pip) | |
| A robot package | `<robot>_rl`, or the name its maker publishes it under | `microduck_rl` |
| The deployment block | `deploy-block-rl-policy`, one for every RL block | |

Every RL repo starts with `learning_block_rl_`, so the blocks and their shared code sort together and show up in one search. The blocks not yet on `ei_rl` keep their old `custom_block_rl_*` names until they move (see [Blocks today](#blocks-today)).

## Vocabulary

The blocks use the frameworks' own terms. In mjlab (and Isaac Lab, whose configuration style mjlab follows):

| Term | What it is | Where it lives (mjlab's Unitree Go1 task) |
|---|---|---|
| **Robot** | Robot config: MJCF, actuators, sensors | A robot module, e.g. mjlab's `asset_zoo/robots/unitree_go1` |
| **Environment** | Environment config: scene, observations, actions, commands, rewards, terminations, randomization, curriculum | `env_cfgs.py` |
| **Policy** | Agent config: network sizes and PPO settings. The trained result (`model.onnx`) is also called the policy | `rl_cfg.py` |
| **Task** | A registered ID that combines an environment and an agent config | `__init__.py`, calling `register_mjlab_task(...)` |

Rewards are part of the environment. A variant of a task ("walk with high steps": the walk with some reward weights changed) is another task, with its own ID.

For Gymnasium and Stable-Baselines3, the same roles are an `Env` class registered with `gym.register` (environment and task) and the PPO hyperparameters (policy).

## The robot package

The robot and its tasks are registered the way the framework does it, with no Edge Impulse imports:

```python
register_mjlab_task(
    task_id="Mjlab-Velocity-Flat-MicroDuck",
    env_cfg=make_microduck_velocity_env_cfg(),
    play_env_cfg=make_microduck_velocity_env_cfg(play=True),
    rl_cfg=MicroduckRlCfg,
)
```

mjlab finds the package without any Edge Impulse code: the package declares an entry point in the `mjlab.tasks` group (`[project.entry-points."mjlab.tasks"]` in its `pyproject.toml`), and mjlab imports every package in that group. The robot's maker, or we, maintain the package; the block pins it to a commit in its Dockerfile and installs it unchanged.

## The shared code: `learning_block_rl_core`

| Module | What it does |
|---|---|
| `ei_rl.core` | The Studio contract: arguments (`parser`, `parse_args`, which ignore the arguments a block doesn't declare), the out and custom data directories, the job's CPUs and memory (`default_workers`, from the cgroup limits), `custom_training_metrics.json`, ONNX helpers, Studio samples. No RL code, and it imports nothing from the rest of `ei_rl` |
| `ei_rl.viz` | The training monitor's format (`rl-viz`): log lines, `manifest.json`, `metrics.json`, `eval.json`, snapshots. Whatever the trainer |
| `ei_rl.bundle` | The policy simulator's format (`ei-rl-sim`): `sim/` (`sim.json`, `scene.mjb.gz`, `visual.bin.gz`), the live checkpoints (`checkpoints/step_<n>.onnx`), `policy/model.onnx` |
| `ei_rl.mjlab` | The mjlab + rsl_rl integration: `RslRlViz` hooks the runner's logger (rsl_rl has no callbacks), logs every PPO iteration, plays the snapshots, saves the checkpoints and stops at the time budget |
| `ei_rl.sb3` | The Gymnasium + Stable-Baselines3 integration: `RLVizCallback`, `export_onnx`, `verify_onnx` |

A block installs it from a pinned commit, with only the extras it uses; there is no PyPI release:

```dockerfile
RUN pip install "ei-rl[mjlab] @ https://github.com/edgeimpulse/learning_block_rl_core/archive/<commit>.tar.gz"
```

The base install needs only numpy and pillow. The extras: `mujoco` for the simulator bundle, `sb3` for Stable-Baselines3 and torch, `mjlab` for mjlab and rsl_rl. A block updates when it moves its pin, and a fix reaches every block that does.

**The formats belong to the extensions.** The training monitor's README specifies `rl-viz` (its owner is Moe Sani), the policy simulator's README specifies `ei-rl-sim`. `ei_rl` implements both and doesn't change them: a new field or file is agreed with the extension's owner and added to the extension first. Both formats carry a `format` and a `version`. A block writes them only through `ei_rl`.

The repo also holds the agent skill for building RL blocks ([skills/ei-build-robotics-rl-blocks](../skills/ei-build-robotics-rl-blocks/SKILL.md)).

## The learn block

The microduck block's files:

| File | What it does |
|---|---|
| `Dockerfile` | Pins microduck_rl and `learning_block_rl_core` to commits and installs them in microduck_rl's own environment |
| `studio.toml` | Which tasks Studio lists, in which order, their labels, how the block evaluates each one, and the reward terms Studio leaves out |
| `tasks/` | The block's own tasks, in the same format as the robot package's (`hop.py`), and the template of Studio's code editor (`_template.py`) |
| `reward_params.py` | Generates `parameters.json` from the tasks (`--check` fails if it is out of date), and applies the reward weights set in Studio |
| `ei_train.py` | The entry point: a training job, see below |
| `rl_viz.py`, `sim_bundle.py` | What is specific to the robot for the extensions: the task descriptor and the snapshot test, the bundle's `sim.json` and its parity check |

`studio.toml` is the only file specific to Edge Impulse around the tasks. It stays in the block, so robot packages don't depend on us:

```toml
hidden_rewards = ["mouth_payload_force"]   # reward terms that are hooks, not settings

[tasks."Mjlab-Velocity-Flat-MicroDuck"]
name = "Walk (velocity tracking), flat ground"
evaluation = "velocity"            # how it's evaluated and published: velocity, standup, ..., episodic
default = true
```

A task in `tasks/` that `studio.toml` doesn't list appears under its ID.

**Studio's form.** The block is self-contained (`"operatesOn": "self_contained_freeform_output"`): the impulse has only the learn block, and the project needs no data. The form has a **Task** option and, under Advanced, the reward weights of the selected task only (`showIf`), generated from the task's own weights. A weight is applied only when it differs from the default, so an untouched form trains the task as registered. The microduck block has 9 tasks and 152 reward weights.

**Your own task in Studio.** The Task option also offers "Your own task (code editor)". The **Task code** parameter (Studio's `code_editor` type) starts with `tasks/_template.py`: a task file that registers one task with its own ID, and may set `STUDIO = dict(name=..., evaluation=...)` in place of a `studio.toml` entry. Its reward weights are in its code, so no weight fields show for it. The block saves it as `policy/task.py`, next to the model.

**Locally**, `python -m tasks train|play|list-envs` runs mjlab's own scripts with the block's tasks registered, so a task is tested with the framework's commands before it is pushed.

## A training job

1. Studio starts the container with the form's values as arguments. `ei_rl.core.parse_args` reads them.
2. The block registers the tasks (the robot package's, its own, the code editor's) and applies the reward weights that differ from their defaults.
3. In the custom data, it clears the previous run's `checkpoints/` and `policy/`, writes the simulation bundle (`sim/`) and announces it in the log, so the policy simulator can run the checkpoints while the job trains. The bundle is optional: a failure there is a warning, not a failed job.
4. It trains through `ei_rl.mjlab.RslRlViz`: every PPO iteration is logged for the training monitor, snapshots (a fixed test of the current policy, as a GIF) play at fixed fractions of the run, checkpoints are exported as ONNX at least every 15 minutes, and training stops at the time budget (`--max-minutes`) whatever the iterations.
5. It exports `model.onnx` from the last checkpoint and checks it against the trained torch policy.
6. It evaluates the **ONNX model** on a fixed benchmark (randomization and noise on, fixed seeds, the same whatever the form's values, so impulses compare), writes `eval.json`, and the headline numbers to `custom_training_metrics.json`.
7. It writes `policy/` (below) to the out directory and copies it to the custom data.

## Outputs

| Where | File | Read by |
|---|---|---|
| Out directory | `model.onnx`: input `"input"` `[1, obs_dim]` float32, output `"output"` `[1, act_dim]`, normalization baked in, the actor only | Studio: the impulse, its conversions (TFLite, C++ SDK), the Deployment tab |
| | `custom_training_metrics.json` | Studio's Experiments tab |
| Custom data (`/home/custom_data`, kept with the learn block, readable while the job runs) | `manifest.json`, `metrics.json`, `eval.json`, `snapshots/` | RL training monitor |
| | `sim/`, `checkpoints/step_<n>.onnx`, `policy/model.onnx` | RL policy simulator |
| | `policy/` | deploy-block-rl-policy |

`policy/` holds what the robot needs besides the model:

- `model.onnx`, and the rsl_rl checkpoint it was exported from (`model_<n>.pt`), to resume, fine-tune or re-export it. Only that one: rsl_rl's other checkpoints stay in the job's work directory, which is gone after the job;
- `params/env.yaml` and `params/agent.yaml`, the configs it trained with, with the reward weights set in Studio;
- `DEPLOY.md`, generated for the run: the task and what it trained with, the commands to play it in simulation and to put it on the robot (with the run's task ID and checkpoint name), the reward weights changed in Studio, the evaluation;
- `task.py`, for a task written in the code editor.

## Deployment

Studio's **Deployment** tab gives a deployment block the model and its conversions, never the custom data.

- **The ONNX policy only:** the **Custom block** deployment type. Its zip has `model.onnx`, `trained.tflite`, `trained.savedmodel.zip`, the C++ SDK, `model-parameters/` and `tflite-model/`.
- **Everything about the policy:** the organization's [deploy-block-rl-policy](https://github.com/edgeimpulse/deploy-block-rl-policy) custom deployment block. It has **Mount learn block under /data** on (`mountLearnBlock`): Studio mounts the learn block's folder at `/data`, and the block zips the Custom block deployment's files with `/data/custom_data/policy/`. It works for any RL block that writes `custom_data/policy/`, so no block needs a deployment block of its own. It needs an Enterprise organization and an impulse with a single learn block.

## Adding to it

| To add | Write | Test it locally with |
|---|---|---|
| A **task** on an existing robot | A registration with a new ID that reuses an environment and agent config, with changed rewards or commands, in the block's `tasks/`. Or the same in Studio's code editor | `python -m tasks train <task-id>` |
| An **environment** | An environment config, usually built from an existing one, plus a task that registers it | `train`, `play` |
| A **policy** | An agent config (network, PPO settings), plus a task that pairs it with an environment | `train` |
| A **robot** | A robot package (MJCF, from MuJoCo Menagerie for many robots; robot config; tasks; the `mjlab.tasks` entry point), then a block that pins it, with its `studio.toml`, `rl_viz.py` and `sim_bundle.py` | `list-envs`, `train`, `play` |

After a change to a task or to `studio.toml`, regenerate `parameters.json` with `reward_params.py`. The microduck README's "Your own task" walks through the hop as an example; the [skill](../skills/ei-build-robotics-rl-blocks/SKILL.md) covers a new block end to end.

## Backends

- **mjlab + rsl_rl, the default.** It uses Isaac Lab's configuration style, which robotics users already know, trains many environments in parallel on a GPU, and robot packages like microduck_rl plug in unchanged. The microduck block needs an NVIDIA GPU.
- **Gymnasium + Stable-Baselines3.** For simple robots that train on CPU in about 40 minutes (SO-101 push, miniAuto football).

## Why it is built this way

- **Framework standards, not our own vocabulary:** tutorials and upstream examples apply as they are.
- **Robots in their own packages:** they work outside Edge Impulse, and their makers can maintain them.
- **One shared repo, pinned to a commit:** one implementation of each format, so the extensions work for any robot without code per robot, and blocks stay small. Without a PyPI release or a compatibility promise, the code can change with the blocks.
- **One block per robot, not one for every robot:** robot licenses differ (CC BY-NC-SA for the microduck, all rights reserved for the TurboPi meshes the miniAuto uses), and a block's image only carries its own robot's dependencies.
- **Not the framework's `train` command alone:** it gives no Studio metrics, telemetry, bundle or policy folder; the shared integrations add them to the framework's own training loop.

## Blocks today

| Block | Stack | GPU | On `ei_rl` | Robot and tasks |
|---|---|---|---|---|
| microduck (`learning_block_rl_microduck`) | mjlab + rsl_rl PPO | Yes | Yes, on the `ei-rl-core` branch | microduck_rl, pinned; the hop and the code editor's template in `tasks/` |
| miniAuto football | Gymnasium + SB3 PPO, self-play | No | Yes, locally; not yet committed or renamed | In the block (`miniauto_soccer_env.py`) |
| SO-101 push cube (`custom_block_rl_so101arm_sim`) | Gymnasium + SB3 PPO | No | Not yet: its own copy of the integration | In the block (`so101_push_env.py`) |
| Unitree Go2 walk (`custom_block_rl_unitree_sim`) | Gymnasium + SB3 PPO | No | Not yet: its own copy of the integration | In the block (`go2_env.py`), robot from MuJoCo Menagerie |

The SB3 blocks also write `deploy_config.json` (the control rate, the action scaling, the default joint positions), in their own code. Their robots and tasks are still inside the blocks; moving them to packages (`so101_rl`, `go2_rl`, `miniauto_rl`) with Gymnasium registration is the next step for them.
