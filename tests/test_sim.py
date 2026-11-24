# Copyright (c) 2026 EdgeImpulse Inc.
# SPDX-License-Identifier: BSD-3-Clause-Clear

"""The simulation contract (ei_rl.sim) and its bundle (python/, sim.json v3), on a toy task (toy_sim)."""

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import mujoco
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from ei_rl import bundle  # noqa: E402
from ei_rl.sim.native import Host, from_bundle  # noqa: E402
from toy_sim import XML  # noqa: E402
from toy_sim.bad import Bad  # noqa: E402
from toy_sim.env import Rail  # noqa: E402
from toy_sim.robot import Cart  # noqa: E402


def episode(host, state=0, steps=30):
    obs = [host.reset(state)["robot"]]
    for k in range(steps):
        o, d = host.step(np.array([np.sin(0.2 * k)]))
        obs.append(o["robot"])
    return np.array(obs), d


class HostTest(unittest.TestCase):
    def setUp(self):
        self.model = mujoco.MjModel.from_xml_string(XML)

    def test_seeded(self):
        a, _ = episode(Host(self.model, Rail, Cart, seed=3))
        b, _ = episode(Host(self.model, Rail, Cart, seed=3))
        c, _ = episode(Host(self.model, Rail, Cart, seed=4))
        np.testing.assert_array_equal(a, b)
        self.assertGreater(np.abs(a - c).max(), 0)

    def test_reset_clears_the_robot(self):
        host = Host(self.model, Rail, Cart, dict(noise=False, randomize=False))
        a, _ = episode(host)
        b, _ = episode(host)
        np.testing.assert_array_equal(a[:, 1], b[:, 1])  # the start jitter differs; the lag must not carry over

    def test_randomize_off_is_nominal(self):
        host = Host(self.model, Rail, Cart, dict(randomize=True))
        host.reset(0)
        drawn = host.model.body_mass[1]
        host.sim.options["randomize"] = False
        host.reset(0)
        self.assertNotEqual(drawn, 1.0)
        self.assertEqual(host.model.body_mass[1], 1.0)

    def test_time_out(self):
        host = Host(self.model, Rail, Cart)
        host.reset(0)
        d = None
        while d is None:
            _, d = host.step(np.zeros(1))
        self.assertEqual(d["reason"], "time_out")
        self.assertEqual(host.steps, round(Rail.episode_s / host.sim.dt))

    def test_action_delay(self):
        now, _ = episode(Host(self.model, Rail, Cart, dict(noise=False, randomize=False)))
        late, _ = episode(Host(self.model, Rail, Cart, dict(noise=False, randomize=False, action_delay=2)))
        np.testing.assert_array_equal(now[0], late[0])
        self.assertGreater(np.abs(now[2:] - late[2:]).max(), 0)

    def test_only_contract_fields(self):
        host = Host(self.model, Rail, Cart)
        with self.assertRaisesRegex(AttributeError, "contract"):
            host.sim.data.efc_force  # noqa: B018
        with self.assertRaises(AttributeError):
            host.sim.data.qpos = np.zeros(1)

    def test_ray(self):
        host = Host(self.model, Rail, Cart)
        host.reset(0)
        dist, geom = host.sim.ray([0, 0, 1], [0, 0, -1], bodyexclude=1)
        self.assertAlmostEqual(dist, 0.99)
        self.assertEqual(host.sim.id2name("geom", geom), "rail")


class BundleTest(unittest.TestCase):
    def test_python_files(self):
        files = bundle.python_files(Rail, Cart)
        self.assertEqual(sorted(f for f in files if not f.startswith("ei_rl/")),
                         ["toy_sim/__init__.py", "toy_sim/env.py", "toy_sim/robot.py"])
        self.assertIn("ei_rl/sim/host.py", files)
        self.assertNotIn("ei_rl/sim/native.py", files)
        self.assertEqual(files["toy_sim/__init__.py"], b"")

    def test_python_files_refuses(self):
        with self.assertRaises(ValueError) as e:
            bundle.python_files(Bad)
        msg = str(e.exception)
        for name in ("gymnasium", "mujoco", "ei_rl.sim.native", "toy_sim.XML"):
            self.assertIn(name, msg)

    def test_round_trip(self):
        model = mujoco.MjModel.from_xml_string(XML)
        spec = bundle.task_spec(Rail, Cart, model)
        self.assertEqual((spec["version"], spec["observation"]["dim"], spec["action"]["dim"]), (3, 4, 1))
        self.assertEqual(spec["python"]["task"], "toy_sim.env:Rail")
        self.assertEqual([s["name"] for s in spec["settings"]], ["lag_s"])
        with tempfile.TemporaryDirectory() as d, contextlib.redirect_stdout(io.StringIO()):
            sim_dir = bundle.write_bundle(d, spec, bundle.model_bytes(model), b"", {},
                                          bundle.python_files(Rail, Cart))
            json.loads((sim_dir / "sim.json").read_text())
            # the bundle's copy, not this directory's
            saved = {k: sys.modules.pop(k) for k in list(sys.modules) if k.split(".")[0] == "toy_sim"}
            try:
                host = from_bundle(sim_dir, seed=5)
                self.assertTrue(sys.modules["toy_sim.env"].__file__.startswith(str(sim_dir)))
                got, _ = episode(host)
            finally:
                for k in [k for k in sys.modules if k.split(".")[0] == "toy_sim"]:
                    del sys.modules[k]
                sys.modules.update(saved)
                sys.path.remove(str(sim_dir / "python"))
        want, _ = episode(Host(model, Rail, Cart, seed=5))
        np.testing.assert_array_equal(got, want)


if __name__ == "__main__":
    unittest.main()
