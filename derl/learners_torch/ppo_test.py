# pylint: disable=missing-docstring
import torch
from derl.env.make_env_torch import make as make_env
from derl.learners_torch.ppo import PPOLearner
from derl.learners_torch.learner_test import LearnerTestCase


class PPOLearnerAtariTest(LearnerTestCase):
  def setUp(self):
    super().setUp()

    kwargs = PPOLearner.get_kwargs()
    self.env = make_env("BreakoutNoFrameskip-v4",
                        nenvs=kwargs.get("nenvs"), seed=0)
    self.learner = PPOLearner.make_with_kwargs(self.env, **kwargs)
    self.learner.model.load_state_dict(
        torch.load("testdata/ppo/atari/model.pt"))
    self.learner.model.to("cpu")

  def test_interactions(self):
    self.assert_interactions("testdata/ppo/atari/interactions.npz",
                             rtol=1e-6, atol=1e-6)

  def test_grad(self):
    self.assert_grad("testdata/ppo/atari/grads.npz", rtol=1e-6, atol=1e-6)

  def test_losses(self):
    self.assert_losses("testdata/ppo/atari/losses.npy", rtol=1e-5, atol=1e-5)


class PPOLearnerPyBulletTest(LearnerTestCase):
  def setUp(self):
    super().setUp()

    kwargs = PPOLearner.get_kwargs("mujoco")
    # Modify some hyper parameters in order for the test not to take to long
    kwargs["num_runner_steps"] = 12
    kwargs["num_minibatches"] = 2
    kwargs["num_epochs"] = 3
    self.env = make_env("HalfCheetahBulletEnv-v0",
                        nenvs=kwargs.get("nenvs"), seed=0)
    self.learner = PPOLearner.make_with_kwargs(self.env, **kwargs)
    self.learner.model.to("cpu")

  def test_interactions(self):
    self.assert_interactions("testdata/ppo/pybullet/interactions.npz",
                             rtol=1e-5, atol=1e-5)

  def test_grad(self):
    self.assert_grad("testdata/ppo/pybullet/grads.npz", rtol=1e-5, atol=1e-5)

  def test_losses(self):
    self.assert_losses("testdata/ppo/pybullet/losses.npy", rtol=1e-5, atol=1e-5)
