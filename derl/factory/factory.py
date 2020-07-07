""" Code to construct different objects. """
from abc import ABC, abstractmethod
from contextlib import contextmanager
from derl.scripts.parsers import get_defaults_parser


class Factory(ABC):
  """ Factory to construct learning algorithms. """
  def __init__(self, **kwargs):
    self.kwargs = kwargs

  @staticmethod
  def get_parser_defaults(args_type="atari"):
    """ Returns default argument dictionary for argument parsing. """
    return {}.get(args_type, {})

  @classmethod
  def get_kwargs(cls, args_type="atari"):
    """ Returns dictionary of keyword arguments. """
    dummy_parser = get_defaults_parser(cls.get_parser_defaults(args_type))
    args = dummy_parser.parse_args([])
    return vars(args)

  @classmethod
  def from_args(cls, args_type="atari", args=None):
    """ Creates factory after parsing command line arguments. """
    defaults = cls.get_parser_defaults(args_type)
    parser = get_defaults_parser(defaults)
    args = parser.parse_args(args)
    return cls(**vars(args))

  def has_arg(self, name):
    """ Returns bool indicating whether the factory has argument with name. """
    return name in self.kwargs

  def get_arg(self, name):
    """ Returns argument value. """
    return self.kwargs[name]

  def get_arg_default(self, name, default=None):
    """ Returns argument value or default if it was not specified. """
    return self.kwargs.get(name, default)

  def get_arg_list(self, *names):
    """ Returns list of argument values. """
    return [self.get_arg(name) for name in names]

  def get_arg_dict(self, *names, check_exists=True):
    """ Returns dictionary of arguments. """
    return {name: self.get_arg(name) for name in names
            if not check_exists or self.has_arg(name)}

  def set_arg(self, **kwargs):
    """ Sets value of keyword argument. """
    for name, val in kwargs.items():
      self.kwargs[name] = val

  @contextmanager
  def custom_kwargs(self, **kwargs):
    """ Custom keyword arguments context manager. """
    init_kwargs = self.kwargs
    self.kwargs = {name: kwargs[name] if name in kwargs else self.kwargs[name]
                   for name in set(self.kwargs) | set(kwargs)}
    try:
      yield
    finally:
      self.kwargs = init_kwargs

  @abstractmethod
  def make_runner(self, env, nlogs=1e5, **kwargs):
    """ Creates and returns algorithm runner. """

  @abstractmethod
  def make_trainer(self, runner, **kwargs):
    """ Creates and returns algorithm trainer. """

  @abstractmethod
  def make_alg(self, runner, trainer, **kwargs):
    """ Creates and returns alg instance with specified runner and trainer. """

  def make(self, env, **kwargs):
    """ Creates and returns algorithm instance. """
    with self.custom_kwargs(**kwargs):
      runner = self.make_runner(env)
      trainer = self.make_trainer(runner)
      alg = self.make_alg(runner, trainer)
      return alg
