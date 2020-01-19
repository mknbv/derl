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

  def learning_generator(self, logdir=None, nlogs=1e5, disable_tqdm=False):
    """ Returns learning generator object. """
    if not getattr(self.runner.step_var, "auto_update", True):
      raise ValueError("learn method is not supported when runner.step_var "
                       "does not auto-update")

    nlogs_has_valid_type = (
        isinstance(nlogs, int)
        or isinstance(nlogs, float) and nlogs.is_integer())
    if not nlogs_has_valid_type:
      raise TypeError(f"nlogs must integer, got {nlogs}")
    if logdir is not None:
      log_period = int(len(self.runner) / nlogs)
      summary.make_writer(logdir)
      summary.record_with_period(log_period, self.runner.step_var)

    with tqdm(total=len(self.runner), disable=disable_tqdm) as pbar:
      for interactions, loss in self.learning_loop():
        pbar.update(int(self.runner.step_var) - pbar.n)
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
