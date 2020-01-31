""" Defines a generic learner. """
from tqdm import tqdm
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
  def make_runner(env, model=None, nlogs=1e5, **kwargs):
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

  def learn(self, disable_tqdm=False):
    """ Performs learning for a specified number of steps. """
    with tqdm(total=len(self.runner), disable=disable_tqdm) as pbar:
      for _ in self.alg.learn():
        pbar.update(self.runner.step_count - pbar.n)
