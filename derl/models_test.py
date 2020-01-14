# pylint: disable=missing-docstring
import collections
from unittest import skipUnless
import numpy as np
import numpy.testing as nt
import torch
from derl.models import (NoisyLinear, NatureCNNBase,
                         NatureCNNModel, MuJoCoModel)
from derl.torch_test_case import TorchTestCase


class NoisyLinearTest(TorchTestCase):
  def test_parameters(self):
    layer = NoisyLinear(3, 4)
    self.assertEqual(len(list(layer.parameters())), 4)

  def test_call(self):
    layer = NoisyLinear(3, 4)
    self.assertEqual(layer(torch.randn(2, 3)).shape, (2, 4))

  @skipUnless(torch.cuda.is_available(), "Requires cuda")
  def test_cuda_call(self):
    layer = NoisyLinear(3, 4)
    layer.to("cuda")
    self.assertEqual(layer(torch.randn(2, 3).to("cuda")).shape, (2, 4))


class DQNBaseTest(TorchTestCase):
  def setUp(self):
    super().setUp()
    self.dqn_base = NatureCNNBase()
    self.dqn_base.load_state_dict(torch.load("testdata/models/dqn-base.pt"))

  def test_params(self):
    self.assertEqual(len(list(self.dqn_base.parameters())), 8)

  def test_noisy_params(self):
    base = NatureCNNBase(noisy=True)
    self.assertEqual(len(list(base.parameters())), 10)

  def test_call(self):
    inputs = torch.rand(32, 84, 84, 4)
    outputs = self.dqn_base(inputs)
    expected = np.load("testdata/models/dqn-base-outputs.npy")
    self.assertAllClose(outputs, expected, atol=1e-6)


def assert_orthogonal(arr):
  """ Checks that np.ndarray has orthogonal initialization. """
  arr = np.reshape(arr, (arr.shape[0], -1))
  nrows, ncols = arr.shape
  if nrows > ncols:
    nt.assert_allclose(arr.T @ arr, np.eye(ncols), atol=1e-6)
  else:
    nt.assert_allclose(arr @ arr.T, np.eye(nrows), atol=1e-6)


class NatureCNNForActorCriticTest(TorchTestCase):
  def setUp(self):
    super().setUp()
    self.dqn = NatureCNNModel(output_units=(4, 1))
    self.dqn.to("cpu")

  def test_params(self):
    nweights = nbiases = 0
    for module in self.dqn.modules():
      if hasattr(module, "bias"):
        nt.assert_equal(module.bias.detach().numpy(), 0.)
        nbiases += 1
      if hasattr(module, "weight"):
        assert_orthogonal(module.weight.detach().numpy())
        nweights += 1
    self.assertEqual(nweights, 6)
    self.assertEqual(nbiases, 6)

  def test_broadcast(self):
    inputs = torch.rand(84, 84, 4)
    outputs = self.dqn(inputs)
    self.assertEqual(len(outputs), 2)
    self.assertEqual(outputs[0].shape, torch.Size((4,)))
    self.assertEqual(outputs[1].shape, torch.Size((1,)))


class NatureCNNForDistributionalRLTest(TorchTestCase):
  def test_output_shape(self):
    dqn = NatureCNNModel(output_units=6, nbins=51)
    outputs = dqn(torch.rand(32, 84, 84, 4))
    self.assertEqual(outputs.shape, torch.Size([32, 6, 51]))


class MuJoCoModelTest(TorchTestCase):
  def test_params(self):
    model = MuJoCoModel(4, 5)
    model.to("cpu")
    nweights = nbiases = 0
    for module in model.modules():
      if hasattr(module, "bias"):
        nt.assert_equal(module.bias.detach().numpy(), 0.)
        nbiases += 1
      if hasattr(module, "weight"):
        assert_orthogonal(module.weight.detach().numpy())
        nweights += 1
    self.assertEqual(nweights, 3)
    self.assertEqual(nbiases, 3)

    # The model should also contain 1 logstd parameter
    self.assertEqual(len(list(model.parameters())), 6 + 1)

  def test_call(self):
    model = MuJoCoModel(4, (5, 1))
    outputs = model(torch.rand(2, 4))
    self.assertIsInstance(outputs, collections.Iterable)
    self.assertEqual(len(outputs), 3)

    mean, std, values = outputs
    self.assertEqual(mean.shape, (2, 5))
    self.assertEqual(std.shape, (2, 5))
    nt.assert_equal(std.cpu().detach().numpy(), 1.)
    self.assertEqual(values.shape, (2, 1))

  def test_broadcast(self):
    model = MuJoCoModel(3, 5)
    outputs = model(torch.rand(3))
    self.assertIsInstance(outputs, collections.Iterable)
    self.assertEqual(len(outputs), 2)
    self.assertEqual(outputs[0].shape, (5,))
    self.assertEqual(outputs[1].shape, (5,))

  def test_dtype(self):
    model = MuJoCoModel(3, 5)
    outputs = model(torch.rand(3).double())
    self.assertEqual(outputs[0].shape, (5,))
    self.assertEqual(outputs[1].shape, (5,))
