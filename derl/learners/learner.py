""" Defines a generic learner. """
import os
from tqdm import tqdm

import torch
from derl.scripts.parsers import get_defaults_parser
import derl.summary as summary


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

  def get_log_period(self, nlogs=1e5):
    """ Converts nlogs to log period. """
    nlogs_has_valid_type = (
        isinstance(nlogs, int)
        or isinstance(nlogs, float) and nlogs.is_integer())
    if not nlogs_has_valid_type:
      raise TypeError(f"nlogs must integer, got {nlogs}")
    return int(self.runner.nsteps / nlogs)

  def setup_summary(self, logdir=None, nlogs=1e5):
    """ Sets up summary writing. """
    log_period = self.get_log_period(nlogs)
    if logdir is not None:
      summary.make_writer(logdir)
      summary.start_recording()

  def learning_generator(self, logdir=None, nlogs=1e5, disable_tqdm=False):
    """ Returns learning generator object. """
    log_period = self.get_log_period(nlogs)
    log_period = float("inf")
    if logdir is not None:
      summary.make_writer(logdir)
      summary.start_recording()

    last_record_step = self.runner.step_count
    with tqdm(total=len(self.runner), disable=disable_tqdm) as pbar:
      for interactions, loss in self.learning_loop():
        pbar.update(int(self.runner.step_count) - pbar.n)

        step_count = self.runner.step_count + 1
        should_record = step_count + 1 - last_record_step >= log_period
        summary.set_recording(should_record)
        if should_record:
          last_record_step = step_count

        yield interactions, loss

  def learn(self, logdir=None, nlogs=1e5, save_weights=None):
    """ Performs learning for a specified number of steps. """
    if save_weights and logdir is None:
      raise ValueError("logdir cannot be None when save_weights is True")
    if save_weights is None:
      save_weights = logdir is not None

    for _ in self.learning_generator(logdir, nlogs):
      pass

    if save_weights:
      torch.save(self.model.state_dict(), os.path.join(logdir, "model"))
