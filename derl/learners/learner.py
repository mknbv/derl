""" Defines a generic learner. """
import os
import tensorflow as tf
from tqdm import tqdm

from derl.train import StepVariable


class Learner:
  """ High-level class for performing learning. """
  def __init__(self, runner, alg):
    self.runner = runner
    self.alg = alg

  @staticmethod
  def get_defaults(env_type="atari"):
    """ Returns default hyperparameters for specified env type. """
    return {}[env_type]

  @staticmethod
  def make_runner(env, args, model=None):
    """ Creates a runner based on the argparse Namespace. """
    raise NotImplementedError("Learner does not implement make_runner method")

  @staticmethod
  def make_alg(runner, args):
    """ Creates learner algorithm. """
    raise NotImplementedError("Learner does not implement make_alg method")

  @property
  def model(self):
    """ Model trained by the algorithm. """
    return self.alg.model

  @classmethod
  def from_env_args(cls, env, args, model=None):
    """ Creates a learner instance from environment and args namespace. """
    runner = cls.make_runner(env, args, model=model)
    return cls(runner, cls.make_alg(runner, args))

  def learning_body(self):
    """ Learning loop body. """
    data = self.runner.get_next()
    loss = self.alg.step(data)
    yield data, loss

  def learning_generator(self, nsteps, logdir=None, log_period=1):
    """ Returns learning generator object. """
    if not getattr(self.runner.step_var, "auto_update", True):
      raise ValueError("learn method is not supported when runner.step_var "
                       "does not auto-update")

    if logdir is not None:
      summary_writer = tf.contrib.summary.create_file_writer(logdir)
      summary_writer.set_as_default()
    step = self.runner.step_var
    if isinstance(step, StepVariable):
      step = step.variable

    with tqdm(total=nsteps) as pbar,\
        tf.contrib.summary.record_summaries_every_n_global_steps(
            log_period, global_step=step):
      while int(self.runner.step_var) < nsteps:
        pbar.update(int(self.runner.step_var) - pbar.n)
        yield from self.learning_body()

  def learn(self, nsteps, logdir=None, log_period=1, save_weights=None):
    """ Performs learning for a specified number of steps. """
    if save_weights and logdir is None:
      raise ValueError("logdir cannot be None when save_weights is True")
    if save_weights is None:
      save_weights = logdir is not None

    for _ in self.learning_generator(nsteps, logdir, log_period):
      pass

    if save_weights:
      self.model.save_weights(os.path.join(logdir, "model"))
