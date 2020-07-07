# pylint: disable=missing-docstring
import torch
from derl.env.make_env import make as make_env
from derl.factory.a2c import A2CFactory
from derl.alg.test import AlgTestCase


class A2CLearnerTest(AlgTestCase):
  def setUp(self):
    super().setUp()

    kwargs = A2CFactory.get_kwargs()
    self.env = make_env("SpaceInvadersNoFrameskip-v4",
                        nenvs=kwargs.get("nenvs"), seed=0)
    self.alg = A2CFactory(**kwargs).make(self.env)
    self.alg.model.load_state_dict(torch.load("testdata/a2c/atari/model.pt"))
    self.alg.model.to("cpu")

  def test_interactions(self):
    self.assert_interactions("testdata/a2c/atari/interactions.npz",
                             rtol=1e-6, atol=1e-6)

  def test_grad(self):
    self.assert_grad("testdata/a2c/atari/grads.npz", rtol=1e-6, atol=1e-6)

  def test_losses(self):
    self.assert_losses("testdata/a2c/atari/losses.npy", rtol=1e-5, atol=1e-4)
