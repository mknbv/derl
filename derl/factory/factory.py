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
  def __init__(self, *, unused_kwargs=None, **kwargs):
    self.kwargs = kwargs
    self.allowed_unused_kwargs = set(unused_kwargs) if unused_kwargs else set()
    self.unused_kwargs = set(self.kwargs)

  @staticmethod
  @abstractmethod
  def get_parser_defaults(args_type="atari"):
    """ Returns default argument dictionary for argument parsing. """

  @classmethod
  def get_kwargs(cls, args_type="atari"):
    """ Returns dictionary of keyword arguments. """
    dummy_parser = get_defaults_parser(cls.get_parser_defaults(args_type))
    args = dummy_parser.parse_args([])
    return vars(args)

  @classmethod
  def with_default_kwargs(cls, args_type="atari", unused_kwargs=None, **kwargs):
    """ Creates instance with default keyword arguments. """
    default_kwargs = cls.get_kwargs(args_type)
    default_kwargs.update(kwargs)
    return cls(unused_kwargs=unused_kwargs, **default_kwargs)

  @classmethod
  def from_args(cls, args_type="atari", unused_kwargs=None, args=None):
    """ Creates factory after parsing command line arguments. """
    defaults = cls.get_parser_defaults(args_type)
    parser = get_defaults_parser(defaults)
    args = parser.parse_args(args)
    return cls(unused_kwargs=unused_kwargs, **vars(args))

  def has_arg(self, name):
    """ Returns bool indicating whether the factory has argument with name. """
    result = name in self.kwargs
    self.unused_kwargs.discard(name)
    return result

  def get_arg(self, name):
    """ Returns argument value. """
    result = self.kwargs[name]
    self.unused_kwargs.discard(name)
    return result

  def get_arg_default(self, name, default=None):
    """ Returns argument value or default if it was not specified. """
    result = self.kwargs.get(name, default)
    self.unused_kwargs.discard(name)
    return result

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
    """ Custom keyword arguments context. """
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

  def make(self, env, nlogs=1e5, check_kwargs=True, **kwargs):
    """ Creates and returns algorithm instance. """
    with self.custom_kwargs(**kwargs):
      runner = self.make_runner(env, nlogs=nlogs)
      trainer = self.make_trainer(runner)
      alg = self.make_alg(runner, trainer)
      if check_kwargs and self.unused_kwargs - self.allowed_unused_kwargs:
        raise ValueError(
            "constructing target object does not use all keyword arguments, "
            "unused keyword arguments are: "
            f"{self.unused_kwargs - self.allowed_unused_kwargs};"
            "if this is expected, consider adding them to unused_kwargs "
            "during factory construction or passing "
            "`check_kwargs=False` to this method.")
    self.unused_kwargs = set(self.kwargs)
    return alg
