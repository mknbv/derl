# pylint: disable=missing-docstring
from unittest import TestCase
import numpy as np
import numpy.testing as nt
import torch
from derl.models_torch import NatureDQNBase, NatureDQN

torch.manual_seed(0)


class DQNBaseTest(TestCase):
  def test_call(self):  # pylint: disable=no-self-use
    inputs = torch.rand(32, 84, 84, 4)
    dqn_base = NatureDQNBase()
    outputs = dqn_base(inputs)
    expected = np.load("testdata/models/dqn-base-outputs.npy")
    nt.assert_allclose(outputs.detach().numpy(), expected)


class NatureDQNForActorCriticTest(TestCase):
  def setUp(self):
    self.dqn = NatureDQN(output_units=(4, 1))

  def test_params(self):
    nweights = nbiases = 0
    for module in self.dqn.modules():
      if hasattr(module, "bias"):
        nt.assert_equal(module.bias.detach().numpy(), 0.)
        nbiases += 1
      if hasattr(module, "weight"):
        weight = module.weight.detach().numpy()
        weight = np.reshape(weight, (weight.shape[0], -1))
        nrows, ncols = weight.shape
        if nrows > ncols:
          nt.assert_allclose(weight.T @ weight, np.eye(ncols), atol=1e-6)
        else:
          nt.assert_allclose(weight @ weight.T, np.eye(nrows), atol=1e-6)
        nweights += 1
    self.assertEqual(nweights, 6)
    self.assertEqual(nbiases, 6)

  def test_broadcast(self):
    inputs = torch.rand(84, 84, 4)
    outputs = self.dqn(inputs)
    self.assertEqual(len(outputs), 2)
    self.assertEqual(outputs[0].shape, torch.Size((4,)))
    self.assertEqual(outputs[1].shape, torch.Size((1,)))


class NatureDQNForDistributionalRLTest(TestCase):
  def setUp(self):
    self.dqn = NatureDQN(output_units=6, nbins=51)
    self.inputs = torch.randn(84, 84, 4)

  def test_output_shape(self):
    outputs = self.dqn(torch.rand(32, 84, 84, 4))
    self.assertEqual(outputs.shape, torch.Size([32, 6, 51]))

  def test_categorical(self):
    dqn = NatureDQN.categorical(output_units=6, nbins=3,
                                val_range=(-1, 1))
    outputs = dqn(self.inputs)
    qvalues = dqn.qvalues_from_outputs(outputs)

    expected = np.sum(outputs.detach().numpy() * np.asarray([-1, 0, 1]), -1)
    nt.assert_allclose(qvalues.detach().numpy(), expected)

  def test_quantile(self):
    dqn = NatureDQN.quantile(output_units=4, nbins=3)
    outputs = dqn(self.inputs)
    qvalues = dqn.qvalues_from_outputs(outputs)
    expected = np.mean(outputs.detach().numpy(), -1)
    nt.assert_allclose(qvalues.detach().numpy(), expected)
