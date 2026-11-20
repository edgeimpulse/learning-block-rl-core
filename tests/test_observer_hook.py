# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""The observer hook helpers in ei_rl.bundle: the export check and the reference runner."""

import math
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
        """A complementary filter on the accelerometer's pitch, the kind of observer a balancer runs."""

        def __init__(self, atan=None):
            super().__init__()
            self.atan = atan or bundle.atan2

        def forward(self, sensors, last_action, state, noise, reset):
            acc_pitch = self.atan(sensors[:, 2:3], -sensors[:, 1:2])
            pitch = torch.where(reset > 0.5, acc_pitch, 0.99 * (state + sensors[:, 3:4] * 0.005) + 0.01 * acc_pitch)
            obs = torch.cat([pitch + 0.01 * noise, sensors[:, 3:4], last_action], dim=1)
            return obs, pitch, (torch.abs(pitch - math.pi / 2) > 0.5).to(torch.float32)

    class Noisy(Filter):
        def forward(self, sensors, last_action, state, noise, reset):
            obs, pitch, t = super().forward(sensors, last_action, state, noise, reset)
            return obs + torch.randn_like(obs), pitch, t


@unittest.skipIf(torch is None, "needs mujoco, torch, onnx and onnxruntime")
class ObserverHookTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.model = mujoco.MjModel.from_xml_string(MJCF)

    def export(self, module):
        return bundle.export_observer(module, self.dir / "observer.onnx", sensor_dim=6, action_dim=2, state_dim=1,
                                      noise_dim=1, terminated=True)

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
        spec = bundle.observer_spec(self.model, "observer.onnx", ["acc", "gyro"], 1,
                                    [{"dist": "normal", "dim": 1, "label": "pitch"}], 4)
        self.assertEqual((spec["noise_dim"], bundle.sensor_dim(self.model, spec["sensors"])), (1, 6))
        with self.assertRaises(KeyError):
            bundle.observer_spec(self.model, "o.onnx", ["nope"], 1, [], 4)
        with self.assertRaises(ValueError):
            bundle.observer_spec(self.model, "o.onnx", ["acc"], 1, [{"dist": "flip", "dim": 1}], 4)

    def test_hook_runs_like_the_module(self):
        path = self.export(Filter())
        spec = bundle.observer_spec(self.model, path.name, ["acc", "gyro"], 1,
                                    [{"dist": "normal", "dim": 1, "label": "pitch"}], 4)
        hook, module = bundle.ObserverHook(self.model, spec, path), Filter()
        data = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, data)
        rng = np.random.default_rng(0)
        state, action = torch.zeros(1, 1), np.zeros(2)
        for i in range(50):
            noise = hook.noise(rng)
            obs = hook.reset(data, 2, noise) if i == 0 else hook.step(data, action, noise)
            sensors = torch.tensor(data.sensordata, dtype=torch.float32)[None]
            want, state, t = module(sensors, torch.tensor(action, dtype=torch.float32)[None], state,
                                    torch.from_numpy(noise)[None], torch.tensor([[float(i == 0)]]))
            np.testing.assert_allclose(obs, want[0].detach().numpy(), atol=1e-5)
            self.assertEqual(hook.terminated, bool(t[0, 0] > 0.5))
            action = rng.uniform(-1, 1, 2)
            mujoco.mj_step(self.model, data)
        self.assertAlmostEqual(float(hook.state[0]), float(state[0, 0]), places=5)


if __name__ == "__main__":
    unittest.main()
