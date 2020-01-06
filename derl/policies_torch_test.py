# pylint: disable=missing-docstring
import numpy as np
import torch
from derl.models_torch import NatureCNNModule
from derl.policies_torch import ActorCriticPolicy, EpsilonGreedyPolicy
from derl.torch_test_case import TorchTestCase


class ActorCriticPolicyTest(TorchTestCase):
  def test_categorical(self):
    model = NatureCNNModule((6, 1))
    policy = ActorCriticPolicy(model)
    act = policy.act(torch.rand(84, 84, 4))
    self.assertEqual(list(act.keys()), ["actions", "log_prob", "values"])
    self.assertEqual(act["actions"], np.array(3))
    self.assertAllClose(act["log_prob"], np.array(-1.8075419664382935))
    self.assertAllClose(act["values"], np.array([0.25730526]))


class EpsilonGreedyPolicyTest(TorchTestCase):
  def act_check(self, policy, expected):
    act = policy.act(torch.randn(84, 84, 4))
    self.assertEqual(list(act.keys()), ["actions"])
    self.assertEqual(act["actions"], expected)

  def test_categorical_model(self):
    model = NatureCNNModule(8, nbins=10)
    policy = EpsilonGreedyPolicy.categorical(model, epsilon=0)
    self.act_check(policy, np.array(5))

  def test_quantile_model(self):
    model = NatureCNNModule(8, nbins=10)
    policy = EpsilonGreedyPolicy.quantile(model, epsilon=0)
    self.act_check(policy, np.array(5))

  def test_dqn(self):
    model = NatureCNNModule(12)
    policy = EpsilonGreedyPolicy(model)
    self.act_check(policy, np.array(2))
