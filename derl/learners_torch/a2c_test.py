# pylint: disable=missing-docstring
import torch
from derl.env.make_env_torch import make as make_env
from derl.learners_torch.a2c import A2CLearner
from derl.learners_torch.learner_test import LearnerTestCase


class A2CLearnerTest(LearnerTestCase):
  def setUp(self):
    super().setUp()

    kwargs = A2CLearner.get_kwargs()
    self.env = make_env("SpaceInvadersNoFrameskip-v4",
                        nenvs=kwargs.get("nenvs"), seed=0)
    self.learner = A2CLearner.make_with_kwargs(self.env, **kwargs)
    self.learner.model.load_state_dict(
        torch.load("testdata/a2c-torch-atari/model.pt"))
    self.learner.model.to("cpu")

  def test_interactions(self):
    self.assert_interactions("testdata/a2c-torch-atari/interactions.npz",
                             rtol=1e-6, atol=1e-6)

  def test_grad(self):
    self.assert_grad("testdata/a2c-torch-atari/grads.npz",
                     rtol=1e-6, atol=1e-6)

  def test_losses(self):
    self.assert_losses("testdata/a2c-torch-atari/losses.npy",
                       rtol=1e-5, atol=1e-4)
