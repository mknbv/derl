""" Defines learner test case. """
import numpy as np
from derl.torch_test_case import TorchTestCase
import derl.summary as summary


class AlgTestCase(TorchTestCase):
  """ Generic learner test case. """
  def setUp(self):
    super().setUp()
    self.env = None
    self.alg = None
    summary.stop_recording()
    summary.should_record = lambda *args, **kwargs: False

  def save_interactions(self, fname):
    """ Saves interactions to a file. """
    interactions = next(self.alg.runner.run())
    np.savez(fname, **interactions)

  def assert_interactions(self, fname, ignore_keys=("state", "infos"),
                          rtol=1e-7, atol=0.):
    """ Checks that interactions have values from the file. """
    ignore_keys = set(ignore_keys) or {}
    interactions = next(self.alg.runner.run())
    with np.load(fname, allow_pickle=True) as expected:
      self.assertEqual(sorted(list(interactions.keys())),
                       sorted(list(expected.keys())))
      for key in filter(lambda k: k not in ignore_keys, expected.keys()):
        with self.subTest(key=key):
          self.assertAllClose(interactions[key], expected[key],
                              rtol=rtol, atol=atol)

  def save_grad(self, fname):
    """ Saves gradient to the file. """
    interactions = next(self.alg.runner.run())
    loss = self.alg.loss(interactions)
    loss.backward()
    grads = {f"grad_{i}": param.grad.numpy() for i, param in
             enumerate(self.alg.model.parameters())}
    np.savez(fname, **grads)

  def assert_grad(self, fname, rtol=1e-7, atol=0.):
    """ Checks that the gradients are close to the values from the file. """
    interactions = next(self.alg.runner.run())
    loss = self.alg.loss(interactions)
    loss.backward()
    with np.load(fname) as expected:
      for i, param in enumerate(self.alg.model.parameters()):
        with self.subTest(grad_i=i):
          self.assertAllClose(param.grad, expected[f"grad_{i}"],
                              rtol=rtol, atol=atol)

  def save_losses(self, filename, num_losses):
    """ Saves losses to the file. """
    data_iter = self.alg.runner.run()
    losses = []
    for _ in range(num_losses):
      losses.append(self.alg.step(next(data_iter)).detach().item())
    np.save(filename, np.asarray(losses))

  def assert_losses(self, filename, rtol=1e-6, atol=0.):
    """ Checks that loss values are close to those from the file. """
    expected = np.load(filename)
    data_iter = self.alg.runner.run()
    for i in range(expected.shape[0]):
      loss = self.alg.step(next(data_iter))
      with self.subTest(i=i):
        self.assertAllClose(loss, expected[i], rtol=rtol, atol=atol)
