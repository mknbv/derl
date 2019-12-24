# pylint: disable=missing-docstring
from derl.env.make_env_torch import make as make_env
from derl.learners.ppo_torch import PPOLearner
from derl.learners.learner_torch_test import LearnerTestCase


class PPOLearnerTest(LearnerTestCase):
  def setUp(self):
    super().setUp()

    kwargs = PPOLearner.get_kwargs()
    self.env = make_env("BreakoutNoFrameskip-v4",
                        nenvs=kwargs.get("nenvs"), seed=0)
    self.learner = PPOLearner.make_with_kwargs(self.env, **kwargs)

  def test_grad(self):
    self.assert_grad("testdata/ppo-torch-atari/grads.npz")

  def test_losses(self):
    self.assert_losses("testdata/ppo-torch-atari/losses.npy")
