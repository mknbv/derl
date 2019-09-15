# pylint: disable=missing-docstring
import random
import tempfile
from types import SimpleNamespace
from unittest import TestCase
import numpy as np
import tensorflow as tf
from derl.learners.ppo import PPOLearner
from derl.env.make_env import make as make_env

tf.enable_eager_execution()


class TestPPOLearner(TestCase):
  def setUp(self):
    random.seed(0)
    np.random.seed(0)
    tf.random.set_random_seed(0)

  def test_atari(self):
    defaults = PPOLearner.get_defaults("atari")
    env = make_env("SpaceInvadersNoFrameskip-v4",
                   nenvs=defaults["nenvs"], seed=0)
    args = SimpleNamespace(**{key.replace("-", "_"): val
                              for key, val in defaults.items()})
    learner = PPOLearner.from_env_args(env, args)

    num_test_steps = 12
    expected = [
        0.03779646381735802,
        0.038728635758161545,
        0.03256126493215561,
        0.02655259147286415,
        0.008041039109230042,
        0.02213088609278202,
        0.024815334007143974,
        0.017860371619462967,
        0.011766642332077026,
        0.0014575012028217316,
        0.010299377143383026,
        0.0044595301151275635,
        0.02526124007999897,
    ]
    logdir = tempfile.gettempdir()
    for step, (_, loss) in enumerate(learner.learning_generator(logdir, 1)):
      with self.subTest(step=step):
        self.assertEqual(float(loss), expected[step])
      if step == num_test_steps:
        break
