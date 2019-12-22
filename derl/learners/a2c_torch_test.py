# pylint: disable=missing-docstring
import numpy as np
from derl.env.make_env_torch import make as make_env
from derl.learners.a2c_torch import A2CLearner
from derl.torch_test_case import TorchTestCase
from derl.train_torch import StepVariable


class A2CLearnerTest(TorchTestCase):
  def setUp(self):
    super().setUp()

    StepVariable.unset_global_step()
    kwargs = A2CLearner.get_kwargs()
    self.env = make_env("SpaceInvadersNoFrameskip-v4",
                        nenvs=kwargs.get("nenvs"), seed=0)
    self.learner = A2CLearner.make_with_kwargs(self.env, **kwargs)

  def test_grad(self):
    interactions = next(self.learner.runner.run())
    loss = self.learner.alg.loss(interactions)
    loss.backward()
    expected = np.load("testdata/a2c-torch-atari/grads.npz")
    for i, param in enumerate(self.learner.model.parameters()):
      with self.subTest(grad_i=i):
        self.assertAllClose(param.grad, expected[f"grad_{i}"])

  def test_losses(self):
    expected = np.load("testdata/a2c-torch-atari/losses.npy")
    for i in range(20):
      _, loss = next(self.learner.learning_loop())
      with self.subTest(i=i):
        self.assertAllClose(loss, expected[i])
