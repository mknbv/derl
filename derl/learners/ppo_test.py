# pylint: disable=missing-docstring
import tensorflow as tf
from derl.learners.learner_test import TestLearner
from derl.learners.ppo import PPOLearner

tf.enable_eager_execution()


class TestPPOLearner(TestLearner):
  def test_atari(self):
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
    self.assert_loss_values(PPOLearner, "SpaceInvadersNoFrameskip-v4", expected)
