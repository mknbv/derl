""" Defines helper methods for argument parsing. """
import argparse
import os


def simple_parser(add_env_id=True, add_logdir=True, log_period=1):
  """ Creates and returns a simple parser. """
  parser = argparse.ArgumentParser()

  def maybe_add(add, *args, **kwargs):
    if add:
      parser.add_argument(*args, **kwargs)

  maybe_add(add_env_id, "--env-id", required=True)
  maybe_add(add_logdir, "--logdir", required=True)
  maybe_add(add_logdir, "--log-period", type=int, default=log_period)
  return parser


def defaults_parser(defaults, base_parser=None):
  """ Adds dictionary of defaults to a parser. """
  if base_parser is None:
    base_parser = argparse.ArgumentParser()
  for key, val in defaults.items():
    if isinstance(val, dict):
      base_parser.add_argument(f"--{key}", **val)
    else:
      base_parser.add_argument(f"--{key}", type=type(val), default=val)
  return base_parser


def get_parser(defaults, add_env_id=True, add_logdir=True, log_period=1):
  """ Returns parser for specified algorithm and env type. """
  return defaults_parser(
      defaults, simple_parser(add_env_id, add_logdir, log_period))


def log_args(args, logdir=None):
  """ Writes `Namespace` of arguments to a text file under logdir directory. """
  if logdir is None:
    logdir = args.logdir
  if not os.path.isdir(logdir):
    os.mkdir(logdir)
  with open(os.path.join(logdir, "args.txt"), 'w') as argsfile:
    for key, val in vars(args).items():
      argsfile.write(f"{key}: {val}\n")


def get_args(defaults, env_id=True, logdir=True, log_period=1, log=None):
  """ Returns parsed arguments. """
  if log and not logdir:
    raise ValueError("logdir must be True when log is True")
  parser = get_parser(defaults, env_id, logdir, log_period)
  args = parser.parse_args()
  if log or log is None and logdir:
    log_args(args)
  return args
