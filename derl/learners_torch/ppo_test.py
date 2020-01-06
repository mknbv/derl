# pylint: disable=missing-docstring
import torch
from derl.env.make_env_torch import make as make_env
from derl.learners_torch.ppo import PPOLearner
from derl.learners_torch.learner_test import LearnerTestCase


class PPOLearnerTest(LearnerTestCase):
  def setUp(self):
    super().setUp()

    kwargs = PPOLearner.get_kwargs()
    self.env = make_env("BreakoutNoFrameskip-v4",
                        nenvs=kwargs.get("nenvs"), seed=0)
    self.learner = PPOLearner.make_with_kwargs(self.env, **kwargs)
    self.learner.model.load_state_dict(
        torch.load("testdata/ppo-torch-atari/model.pt"))
    self.learner.model.to("cpu")

  def test_interactions(self):
    self.assert_interactions("testdata/ppo-torch-atari/interactions.npz",
                             rtol=1e-6, atol=1e-6)

  def test_grad(self):
    self.assert_grad("testdata/ppo-torch-atari/grads.npz",
                     rtol=1e-6, atol=1e-6)

  def test_losses(self):
    self.assert_losses("testdata/ppo-torch-atari/losses.npy",
                       rtol=1e-5, atol=1e-5)
