""" Defines learner test case. """
import numpy as np
from derl.torch_test_case import TorchTestCase
from derl.train_torch import StepVariable
import derl.summary as summary


class LearnerTestCase(TorchTestCase):
  """ Generic learner test case. """
  def setUp(self):
    super().setUp()
    StepVariable.unset_global_step()
    self.env = None
    self.learner = None
    summary.stop_recording()

  def assert_grad(self, grads_file):
    """ Checks that the gradients are close to the values from the file. """
    interactions = next(self.learner.runner.run())
    loss = self.learner.alg.loss(interactions)
    loss.backward()
    expected = np.load(grads_file)
    for i, param in enumerate(self.learner.model.parameters()):
      with self.subTest(grad_i=i):
        self.assertAllClose(param.grad, expected[f"grad_{i}"])

  def assert_losses(self, losses_file):
    """ Checks that loss values are close to those from the file. """
    expected = np.load(losses_file)
    for i in range(expected.shape[0]):
      _, loss = next(self.learner.learning_loop())
      with self.subTest(i=i):
        self.assertAllClose(loss, expected[i])
