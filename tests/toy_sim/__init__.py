"""A toy task for the simulation contract's tests: a cart on a rail. Like a block's package, its __init__
imports what the browser lacks; the bundle leaves it empty."""
import mujoco  # noqa: F401

XML = """
<mujoco>
  <option timestep="0.005"/>
  <worldbody>
    <geom name="rail" type="box" size="2 0.05 0.01"/>
    <body name="cart" pos="0 0 0.1">
      <joint name="slide" type="slide" axis="1 0 0" damping="0.5"/>
      <geom name="cart" type="box" size="0.1 0.1 0.05" mass="1"/>
      <site name="tip" pos="0.1 0 0"/>
    </body>
  </worldbody>
  <actuator><motor name="push" joint="slide" gear="5"/></actuator>
</mujoco>
"""
