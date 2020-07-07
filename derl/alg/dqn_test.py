# pylint: disable=missing-docstring
from derl.env.make_env import make as make_env
from derl.factory.dqn import DQNFactory
from derl.alg.test import AlgTestCase


class DQNLearnerTest(AlgTestCase):
  def setUp(self):
    super().setUp()

    kwargs = DQNFactory.get_kwargs()
    kwargs["storage_init_size"] = 42
    self.env = make_env("SpaceInvadersNoFrameskip-v4",
                        nenvs=kwargs.get("nenvs"), seed=0)
    self.alg = DQNFactory(**kwargs).make(self.env)
    self.alg.model.to("cpu")

  def test_interactions(self):
    _ = next(self.alg.runner.run())
