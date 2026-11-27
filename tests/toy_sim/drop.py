"""The second toy, for the fields beyond the kinematics: a box dropped on the floor (contacts, a varying number
of constraint rows) and an arm whose actuator sets its friction from the joint's load, as a load-dependent
servo model does (efc_*, qfrc_bias, qfrc_constraint, a model field written every physics step), with a counter
in data.userdata (a data field written back)."""

import numpy as np

from ei_rl.sim import Robot, Task

FRICTION_DOF = 1  # mjtConstraint.mjCNSTR_FRICTION_DOF

XML = """
<mujoco>
  <option timestep="0.002"/>
  <size nuserdata="2"/>
  <worldbody>
    <geom name="floor" type="plane" size="2 2 0.1"/>
    <body name="box" pos="0 0 0.3"><freejoint/><geom type="box" size="0.05 0.05 0.05"/></body>
    <body name="arm" pos="0.4 0 0.15">
      <joint name="hinge" type="hinge" axis="0 1 0" frictionloss="0.05" range="-1.2 1.2"/>
      <geom type="capsule" fromto="0 0 0 0.2 0 0" size="0.02"/>
    </body>
  </worldbody>
  <actuator><motor name="m" joint="hinge" gear="1"/></actuator>
</mujoco>
"""


class Arm(Robot):
    action_dim = 1

    def __init__(self, sim, name):
        super().__init__(sim, name)
        self.dof = sim.model.jnt_dofadr[sim.id("joint", "hinge")]
        self.act = sim.id("actuator", "m")

    def reset(self, sim, rng):
        self.target = 0.0

    def command(self, sim, action):
        self.target = float(action[0])

    def apply(self, sim):
        d, n = sim.data, sim.data.nefc
        rows = d.efc_type[:n] == FRICTION_DOF
        fric = np.zeros(sim.model.nv)
        np.add.at(fric, d.efc_id[:n][rows], d.efc_force[:n][rows])
        load = -d.qfrc_bias[self.dof] + d.qfrc_constraint[self.dof] - fric[self.dof]  # the rest of the world's
        sim.model.dof_frictionloss[self.dof] = 0.02 + 0.3 * abs(load)
        d.ctrl[self.act] = 0.5 * self.target
        d.userdata[0] += 1.0
        d.userdata[1] = load


class Drop(Task):
    id = "Toy-Drop-v0"
    label = "Toy drop"
    decimation = 5

    def reset(self, sim, rng, state=None):
        sim.data.qpos[:3] = [0.0 if rng is None else rng.uniform(-0.05, 0.05), 0.0, 0.3]

    def observe(self, sim, name, rng):
        d, c = sim.data, sim.data.contact
        return np.concatenate([[d.ncon, d.nefc, d.efc_force.sum(), c.dist.sum(), c.geom.sum(), c.frame.sum(), d.time],
                               d.qfrc_bias, d.qfrc_constraint, d.userdata,
                               sim.model.dof_frictionloss]).astype(np.float32)
