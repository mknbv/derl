""" Defines learner test case. """
import numpy as np
from derl.torch_test_case import TorchTestCase
import derl.summary as summary


class LearnerTestCase(TorchTestCase):
  """ Generic learner test case. """
  def setUp(self):
    super().setUp()
    self.env = None
    self.learner = None
    summary.stop_recording()
    summary.should_record = lambda *args, **kwargs: False

  def assert_interactions(self, fname, ignore_keys=("state", "infos"),
                          rtol=1e-7, atol=0.):
    """ Checks that interactions have values from the file. """
    ignore_keys = set(ignore_keys) or {}
    interactions = next(self.learner.runner.run())
    with np.load(fname, allow_pickle=True) as expected:
      self.assertEqual(sorted(list(interactions.keys())),
                       sorted(list(expected.keys())))
      for key in filter(lambda k: k not in ignore_keys, expected.keys()):
        with self.subTest(key=key):
          self.assertAllClose(interactions[key], expected[key],
                              rtol=rtol, atol=atol)

  def assert_grad(self, fname, rtol=1e-7, atol=0.):
    """ Checks that the gradients are close to the values from the file. """
    interactions = next(self.learner.runner.run())
    loss = self.learner.alg.loss(interactions)
    loss.backward()
    expected = np.load(fname)
    for i, param in enumerate(self.learner.model.parameters()):
      with self.subTest(grad_i=i):
        self.assertAllClose(param.grad, expected[f"grad_{i}"],
                            rtol=rtol, atol=atol)

  def save_losses(self, filename, *, num):
    """ Saves losses to file with specified name. """
    losses = []
    learning_iter = self.learner.alg.learn()
    for _ in range(num):
      _, loss = next(learning_iter)
      losses.append(loss.detach().numpy())
    np.save(filename, np.asarray(losses))

  def assert_losses(self, filename, rtol=1e-6, atol=0.):
    """ Checks that loss values are close to those from the file. """
    expected = np.load(filename)
    learning_iter = self.learner.alg.learn()
    for i in range(expected.shape[0]):
      _, loss = next(learning_iter)
      with self.subTest(i=i):
        self.assertAllClose(loss, expected[i], rtol=rtol, atol=atol)
