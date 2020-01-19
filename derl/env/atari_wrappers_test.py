# pylint: disable=missing-docstring
from unittest import TestCase
import numpy as np
import numpy.testing as nt
from gym import make as gym_make
from derl.env.make_env import make as derl_make


def run_env(env, nsteps=100):
  obs = [env.reset()]
  for _ in range(nsteps):
    action = env.action_space.sample()
    obs.append(env.step(action)[0])
  return np.asarray(obs)


class TestSeed(TestCase):
  # pylint: disable=invalid-name, no-self-use
  def assertArrayEqual(self, expected, actual):
    return nt.assert_array_equal(expected, actual)

  def test_seed(self):
    def make_with_gym():
      env = gym_make("BreakoutNoFrameskip-v4")
      env.seed(0)
      env.action_space.np_random.seed(0)
      return env

    def make_with_derl():
      return derl_make("BreakoutNoFrameskip-v4")

    for makefn in [make_with_gym, make_with_derl]:
      with self.subTest(makefn=makefn.__name__):
        env = makefn()
        obs = run_env(env)
        newenv = makefn()
        newobs = run_env(newenv)

        self.assertArrayEqual(obs, newobs)
