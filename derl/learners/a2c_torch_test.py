# pylint: disable=missing-docstring
from derl.env.make_env_torch import make as make_env
from derl.learners.a2c_torch import A2CLearner
from derl.learners.learner_torch_test import LearnerTestCase


class A2CLearnerTest(LearnerTestCase):
  def setUp(self):
    super().setUp()

    kwargs = A2CLearner.get_kwargs()
    self.env = make_env("SpaceInvadersNoFrameskip-v4",
                        nenvs=kwargs.get("nenvs"), seed=0)
    self.learner = A2CLearner.make_with_kwargs(self.env, **kwargs)

  def test_grad(self):
    self.assert_grad("testdata/a2c-torch-atari/grads.npz")

  def test_losses(self):
    self.assert_losses("testdata/a2c-torch-atari/losses.npy")
