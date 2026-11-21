# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""The observer hook helpers in ei_rl.bundle: the export check and the reference runner."""

import tempfile
import unittest
from pathlib import Path

import numpy as np

try:
    import mujoco
    import onnxruntime  # noqa: F401
    import torch

    from ei_rl import bundle
except ImportError:  # the hook needs the mujoco and sb3 extras
    torch = None

MJCF = """<mujoco><option timestep="0.005"/><worldbody><geom type="plane" size="1 1 .1"/>
<body name="b" pos="0 0 .1"><freejoint/><geom type="box" size=".05 .05 .05" mass=".3"/><site name="imu"/></body>
</worldbody><sensor><accelerometer name="acc" site="imu"/><gyro name="gyro" site="imu"/></sensor></mujoco>"""


if torch is not None:
    class Filter(torch.nn.Module):
        """A complementary filter on the accelerometer's pitch, started from it: state = [pitch, started]."""

        def __init__(self, atan=None):
            super().__init__()
            self.atan = atan or bundle.atan2

        def forward(self, sensors, last_action, state):
            acc_pitch = self.atan(sensors[:, 2:3], -sensors[:, 1:2])
            pitch = 0.99 * (state[:, 0:1] + sensors[:, 3:4] * 0.005) + 0.01 * acc_pitch
            pitch = torch.where(state[:, 1:2] > 0.5, pitch, acc_pitch)
            return torch.cat([pitch, sensors[:, 3:4], last_action], dim=1), torch.cat([pitch, torch.ones_like(pitch)], 1)

    class Noisy(Filter):
        def forward(self, sensors, last_action, state):
            obs, state = super().forward(sensors, last_action, state)
            return obs + torch.randn_like(obs), state


@unittest.skipIf(torch is None, "needs mujoco, torch, onnx and onnxruntime")
class ObserverHookTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.model = mujoco.MjModel.from_xml_string(MJCF)

    def export(self, module):
        return bundle.export_observer(module, self.dir / "observer.onnx", sensor_dim=6, action_dim=2, state_dim=2)

    def test_atan2_matches_numpy(self):
        v = [0.0, -0.0, 1.0, -1.0, 0.3, -2.5, 1e-8]
        y, x = (np.array(a, np.float32) for a in zip(*[(a, b) for a in v for b in v]))
        got = bundle.atan2(torch.from_numpy(y), torch.from_numpy(x)).numpy()
        want = np.arctan2(np.where(y == 0, 0.0, y), np.where(x == 0, 0.0, x))  # -0 counts as 0, in y too (pi, not -pi)
        np.testing.assert_allclose(got, want, atol=3e-7)

    def test_export_checks_the_graph(self):
        self.export(Filter())
        with self.assertRaisesRegex(ValueError, "differs from torch"):
            self.export(Filter(atan=torch.atan2))  # torch's export is wrong at x = -0
        with self.assertRaisesRegex(ValueError, "Random"):
            self.export(Noisy())

    def test_spec(self):
        spec = bundle.observer_spec(self.model, "observer.onnx", ["acc", "gyro"], 2, 4)
        self.assertEqual(spec, {"file": "observer.onnx", "sensors": ["acc", "gyro"], "state_dim": 2, "obs_dim": 4})
        with self.assertRaises(KeyError):
            bundle.observer_spec(self.model, "o.onnx", ["nope"], 2, 4)

    def test_bundle_ships_the_graph(self):
        path = self.export(Filter())
        spec = {"observer": bundle.observer_spec(self.model, path.name, ["acc", "gyro"], 2, 4)}
        sim = bundle.write_bundle(self.dir / "out", spec, b"mjb", b"vis", {}, files={path.name: path.read_bytes()})
        self.assertEqual((sim / path.name).read_bytes(), path.read_bytes())

    def test_hook_runs_like_the_module(self):
        path = self.export(Filter())
        spec = bundle.observer_spec(self.model, path.name, ["acc", "gyro"], 2, 4)
        hook, module = bundle.ObserverHook(self.model, spec, path), Filter()
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        rng = np.random.default_rng(0)
        state, action = torch.zeros(1, 2), np.zeros(2)
        for i in range(50):
            obs = hook.reset(data, 2) if i == 0 else hook.step(data, action)
            sensors = torch.tensor(data.sensordata, dtype=torch.float32)[None]
            want, state = module(sensors, torch.tensor(action, dtype=torch.float32)[None], state)
            np.testing.assert_allclose(obs, want[0].detach().numpy(), atol=1e-5)
            action = rng.uniform(-1, 1, 2)
            mujoco.mj_step(self.model, data)
        np.testing.assert_allclose(hook.state, state[0].numpy(), atol=1e-5)


if __name__ == "__main__":
    unittest.main()
