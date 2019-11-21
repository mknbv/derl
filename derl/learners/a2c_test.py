# pylint: disable=missing-docstring
import tensorflow as tf
from derl.learners.a2c import A2CLearner
from derl.learners.learner_test import TestLearner

tf.enable_eager_execution()


class TestA2CLearner(TestLearner):
  def test_atari(self):
    expected = [
        -0.01737835630774498,
        -0.002556178718805313,
        -0.02713838964700699,
        -0.01362221036106348,
        -0.023149555549025536,
        -0.009293952025473118,
        -0.027666717767715454,
        -0.008767497725784779,
        -0.029115848243236542,
        -0.0005100502166897058,
        0.13843855261802673,
        -0.03295762091875076,
        0.031341858208179474,
    ]
    self.assert_loss_values(A2CLearner, "BreakoutNoFrameskip-v4", expected)
