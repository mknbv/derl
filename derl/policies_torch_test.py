# pylint: disable=missing-docstring
from unittest import TestCase
import numpy as np
import numpy.testing as nt
import torch
from derl.models_torch import NatureDQN
from derl.policies_torch import ActorCriticPolicy, EpsilonGreedyPolicy

torch.manual_seed(0)
np.random.seed(0)


class ActorCriticPolicyTest(TestCase):
  def test_categorical(self):
    model = NatureDQN((6, 1))
    policy = ActorCriticPolicy(model)
    act = policy.act(torch.rand(84, 84, 4))
    self.assertEqual(list(act.keys()), ["actions", "log_prob", "values"])
    self.assertEqual(act["actions"], np.array(5))
    nt.assert_allclose(act["log_prob"], np.array(-1.7288962602615356))
    nt.assert_allclose(act["values"], np.array([0.13464875519275665]))


class EpsilonGreedyPolicyTest(TestCase):
  def act_check(self, policy, expected):
    act = policy.act(torch.randn(84, 84, 4))
    self.assertEqual(list(act.keys()), ["actions"])
    self.assertEqual(act["actions"], expected)

  def test_categorical_model(self):
    model = NatureDQN.categorical(8, nbins=10)
    policy = EpsilonGreedyPolicy(model)
    self.act_check(policy, np.array(1))

  def test_quantile_model(self):
    model = NatureDQN.quantile(8, nbins=10)
    policy = EpsilonGreedyPolicy(model)
    self.act_check(policy, np.array(5))

  def test_dqn(self):
    model = NatureDQN(12)
    policy = EpsilonGreedyPolicy(model)
    self.act_check(policy, np.array(10))
