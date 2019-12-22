""" Defies test case for testing models written in pytorch. """
import random
from unittest import TestCase
import numpy as np
import numpy.testing as nt
import torch


def _np(tensor):
  """ Converts tensor to numpy array. """
  if isinstance(tensor, torch.Tensor):
    return tensor.detach().numpy()
  return tensor



class TorchTestCase(TestCase):
  """ Test case for testing code with models written in pytorch. """
  def setUp(self):
    torch.manual_seed(0)
    random.seed(0)
    np.random.seed(0)

  def assertAllClose(self, actual, expected):  # pylint: disable=invalid-name
    """ Checks that actual and expected arrays or torch tensors are equal. """
    # pylint: disable=no-self-use
    nt.assert_allclose(_np(actual), _np(expected))
