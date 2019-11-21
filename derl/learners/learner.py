""" Defines a generic learner. """
import os
import tensorflow as tf
from tqdm import tqdm

from derl.train import StepVariable
from derl.scripts.parsers import get_defaults_parser


class Learner:
  """ High-level class for performing learning. """
  def __init__(self, runner, alg):
    self.runner = runner
    self.alg = alg

  @classmethod
  def get_parser_defaults(cls, env_type="atari"):
    """ Returns defaults for argument parsing. """
    return {}[env_type]

  @classmethod
  def get_kwargs(cls, env_type="atari"):
    """ Returns kwargs dict with default hyperparameters. """
    dummy_parser = get_defaults_parser(cls.get_parser_defaults(env_type))
    args = dummy_parser.parse_args([])
    return vars(args)

  @staticmethod
  def make_runner(env, model=None, **kwargs):
    """ Creates a runner based on the argparse Namespace. """
    raise NotImplementedError("Learner does not implement make_runner method")

  @staticmethod
  def make_alg(runner, **kwargs):
    """ Creates learner algorithm. """
    raise NotImplementedError("Learner does not implement make_alg method")

  @property
  def model(self):
    """ Model trained by the algorithm. """
    return self.alg.model

  @classmethod
  def make_with_args(cls, env, args, model=None):
    """ Creates a learner instance from environment and args namespace. """
    return cls.make_with_kwargs(env, model=model, **vars(args))

  @classmethod
  def make_with_kwargs(cls, env, model=None, **kwargs):
    """ Creates a learner instance from environment and keyword args. """
    runner = cls.make_runner(env, model=model, **kwargs)
    return cls(runner, cls.make_alg(runner, **kwargs))

  def learning_loop(self):
    """ Learning loop of the learner. """
    for interactions in self.runner.run():
      yield interactions, self.alg.step(interactions)

  def learning_generator(self, logdir=None, log_freq=1e-5, disable_tqdm=False):
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

    if not 0 <= log_freq <= 1:
      raise ValueError(f"log_freq must be in [0, 1], got {log_freq}")
    log_period = int(len(self.runner) * log_freq)
    last_record_step = tf.Variable(step - log_period)

    learning_iterator = self.learning_loop()
    with tqdm(total=len(self.runner), disable=disable_tqdm) as pbar:
      while True:
        summary_context = tf.contrib.summary.never_record_summaries
        should_record_summaries = step - last_record_step >= log_period
        old_step = tf.Variable(step)
        if should_record_summaries:
          summary_context = tf.contrib.summary.always_record_summaries
        try:
          with summary_context():
            step_result = next(learning_iterator)
          if should_record_summaries and step > old_step:
            last_record_step.assign(step)
          pbar.update(int(self.runner.step_var) - pbar.n)
          yield step_result
        except StopIteration:
          break

  def learn(self, logdir=None, log_freq=1e-5, save_weights=None):
    """ Performs learning for a specified number of steps. """
    if save_weights and logdir is None:
      raise ValueError("logdir cannot be None when save_weights is True")
    if save_weights is None:
      save_weights = logdir is not None

    for _ in self.learning_generator(logdir, log_freq):
      pass

    if save_weights:
      self.model.save_weights(os.path.join(logdir, "model"))
