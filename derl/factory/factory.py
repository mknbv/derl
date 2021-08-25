""" Code to construct different objects. """
from abc import ABC, abstractmethod
from contextlib import contextmanager
from derl.scripts.parsers import get_defaults_parser


class KwargsDict:
  """ Key-word arguments dictionary. """
  def __init__(self, **kwargs):
    self.kwargs = kwargs
    self.unused = set(self.kwargs)

  def has_arg(self, key):
    """ Return true if this kwargs dict has key. """
    self.unused.discard(key)
    return key in self.kwargs

  def get_arg(self, key):
    """ Returns argument under key. """
    self.unused.discard(key)
    return self.kwargs[key]

  def get_arg_default(self, key, default=None):
    """ Returns argument under key if it was specified or default otherwise. """
    if key not in self.kwargs:
      return default
    return self.get_arg(key)

  def get_arg_list(self, *keys):
    """ Returns list of arguments under specified keys. """
    return [self.get_arg(key) for key in keys]

  def get_arg_dict(self, *keys, check_exists=True):
    """ Returns dictionary of arugments under specified keys. """
    return {key: self.get_arg(key) for key in keys
            if not check_exists or self.has_arg(key)}

  @contextmanager
  def override_context(self, **kwargs):
    """ Context manager for overriding kwargs. """
    init_kwargs = dict(self.kwargs)
    for key, val in kwargs.items():
      self.kwargs[key] = val
      self.unused.add(key)
    try:
      yield
    finally:
      custom_unused = set(self.unused) & set(kwargs)
      if custom_unused:
        raise ValueError("not all custom kwargs were used in this context, "
                         f"unused kwargs are {custom_unused}")
      self.kwargs = init_kwargs

  def reset_unused(self):
    """ Adds all kwargs to the unused collection. """
    self.unused = set(self.kwargs)


class Factory(ABC):
  """ Factory to construct learning algorithms. """
  def __init__(self, *, ignore_unused=None, **kwargs):
    self.kwargs = KwargsDict(**kwargs)
    self.ignore_unused = set(ignore_unused) if ignore_unused else set()

  def __getattr__(self, name):
    return getattr(self.kwargs, name)

  @staticmethod
  @abstractmethod
  def get_parser_defaults(args_type="atari"):
    """ Returns default argument dictionary for argument parsing. """

  @staticmethod
  def make_env_kwargs(env_id):
    """ Returns keyword arguments for derl.env.make function. """
    _ = env_id
    return {}

  @classmethod
  def get_kwargs(cls, args_type="atari"):
    """ Returns dictionary of keyword arguments. """
    dummy_parser = get_defaults_parser(cls.get_parser_defaults(args_type))
    args = dummy_parser.parse_args([])
    return vars(args)

  @classmethod
  def from_default_kwargs(cls, args_type="atari", ignore_unused=None, **kwargs):
    """ Creates instance with default keyword arguments. """
    default_kwargs = cls.get_kwargs(args_type)
    default_kwargs.update(kwargs)
    return cls(ignore_unused=ignore_unused, **default_kwargs)

  @classmethod
  def from_args(cls, args_type="atari", ignore_unused=None, args=None):
    """ Creates factory after parsing command line arguments. """
    defaults = cls.get_parser_defaults(args_type)
    parser = get_defaults_parser(defaults)
    args = parser.parse_args(args)
    return cls(ignore_unused=ignore_unused, **vars(args))

  @abstractmethod
  def make_runner(self, env, nlogs=1e5, **kwargs):
    """ Creates and returns algorithm runner. """

  @abstractmethod
  def make_trainer(self, runner, **kwargs):
    """ Creates and returns algorithm trainer. """

  @abstractmethod
  def make_alg(self, runner, trainer, **kwargs):
    """ Creates and returns alg instance with specified runner and trainer. """

  def make(self, env, nlogs=1e5, check_kwargs=True, **kwargs):
    """ Creates and returns algorithm instance. """
    with self.override_context(**kwargs):
      runner = self.make_runner(env, nlogs=nlogs)
      trainer = self.make_trainer(runner)
      alg = self.make_alg(runner, trainer)
      if check_kwargs and self.kwargs.unused - self.ignore_unused:
        raise ValueError(
            "constructing target object does not use all keyword arguments, "
            "unused keyword arguments are: "
            f"{self.kwargs.unused - self.ignore_unused};"
            "if this is expected, consider adding them to ignore_unused "
            "during factory construction or passing "
            "`check_kwargs=False` to this method.")
    self.kwargs.reset_unused()
    return alg
