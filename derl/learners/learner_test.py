""" Implements TestLearner class. """
import random
import tempfile
import numpy as np
import tensorflow as tf
from derl.env.make_env import make as make_env, is_atari_id, is_mujoco_id


def get_env_type(env_id):
  """ Returns environment type for a given environment id. """
  if is_atari_id(env_id):
    return "atari"
  if is_mujoco_id(env_id):
    return "mujoco"
  raise ValueError(f"env_id {env_id} has unknown type")


class TestLearner(tf.test.TestCase):
  """ General test case to be inherited from to test a specific learner. """
  def setUp(self):
    tf.reset_default_graph()
    random.seed(0)
    np.random.seed(0)
    tf.random.set_random_seed(0)

  def assert_loss_values(self, learner_class, env_id, loss_values):
    """ Checks that learner has specified loss values. """
    env_type = get_env_type(env_id)
    kwargs = learner_class.get_kwargs(env_type)
    env = make_env(env_id, nenvs=kwargs["nenvs"], seed=0)
    learner = learner_class.make_with_kwargs(env, **kwargs)

    logdir = tempfile.gettempdir()
    for step, (_, loss) in enumerate(
        learner.learning_generator(logdir, 1, disable_tqdm=True)):
      with self.subTest(step=step):
        self.assertEqual(float(loss), loss_values[step])
      if step + 1 == len(loss_values):
        break
