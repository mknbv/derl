# pylint: disable=missing-docstring
import numpy as np
import torch
from derl.models import NatureCNNModel, MuJoCoModel
from derl.policies import ActorCriticPolicy, EpsilonGreedyPolicy
from derl.torch_test_case import TorchTestCase


class ActorCriticPolicyTest(TorchTestCase):
  def test_categorical(self):
    model = NatureCNNModel((6, 1))
    model.to("cpu")
    policy = ActorCriticPolicy(model)
    act = policy.act(torch.rand(84, 84, 4))
    self.assertEqual(list(act.keys()), ["actions", "log_prob", "values"])
    self.assertEqual(act["actions"], np.array(3))
    self.assertAllClose(act["log_prob"], np.array(-1.80754196), rtol=1e-6)
    self.assertAllClose(act["values"], np.array([0.257305294]), rtol=1e-6)

  def test_normal(self):
    model = MuJoCoModel(3, (2, 1))
    model.to("cpu")
    policy = ActorCriticPolicy(model)
    act = policy.act(torch.randn(3))
    self.assertEqual(list(act.keys()), ["actions", "log_prob", "values"])
    self.assertAllClose(act["actions"], np.array([-1.7938228, 1.0464325]),
                        rtol=1e-6)
    self.assertAllClose(act["log_prob"], np.array(-3.7467263))
    self.assertAllClose(act["values"], np.array([-0.18482158]), rtol=1e-6)


class EpsilonGreedyPolicyTest(TorchTestCase):
  def act_check(self, policy, expected):
    act = policy.act(torch.randn(84, 84, 4))
    self.assertEqual(list(act.keys()), ["actions"])
    self.assertEqual(act["actions"], expected)

  def test_categorical_model(self):
    model = NatureCNNModel(8, nbins=10)
    policy = EpsilonGreedyPolicy.categorical(model, epsilon=0)
    self.act_check(policy, np.array(5))

  def test_quantile_model(self):
    model = NatureCNNModel(8, nbins=10)
    policy = EpsilonGreedyPolicy.quantile(model, epsilon=0)
    self.act_check(policy, np.array(5))

  def test_dqn(self):
    model = NatureCNNModel(12)
    policy = EpsilonGreedyPolicy(model)
    self.act_check(policy, np.array(2))
