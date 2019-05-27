""" Defines helper methods for argument parsing. """
import argparse
import os
from derl.env import is_atari_id, is_mujoco_id


def get_simple_parser(add_env_id=True, add_logdir=True, log_period=1):
  """ Creates and returns a simple parser. """
  parser = argparse.ArgumentParser()

  def maybe_add(add, *args, **kwargs):
    if add:
      parser.add_argument(*args, **kwargs)

  maybe_add(add_env_id, "--env-id", required=True)
  maybe_add(add_logdir, "--logdir", required=True)
  maybe_add(add_logdir, "--log-period", type=int, default=log_period)
  return parser


def get_defaults_parser(defaults, base_parser=None):
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
  """ Returns parser for specified defaults and env type. """
  return get_defaults_parser(
      defaults, get_simple_parser(add_env_id, add_logdir, log_period))


def log_args(args, logdir=None):
  """ Writes `Namespace` of arguments to a text file under logdir directory. """
  if logdir is None:
    logdir = args.logdir
  if not os.path.isdir(logdir):
    os.makedirs(logdir, exist_ok=True)
  with open(os.path.join(logdir, "args.txt"), 'w') as argsfile:
    for key, val in vars(args).items():
      argsfile.write(f"{key}: {val}\n")
  return args


def get_args_from_defaults(defaults, env_id=True,
                           logdir=True, log_period=1, call_log_args=None):
  """ Returns parsed arguments. """
  if call_log_args and not logdir:
    raise ValueError("logdir must be True when call_log_args is True")
  parser = get_parser(defaults, env_id, logdir, log_period)
  args = parser.parse_args()
  if call_log_args or call_log_args is None and logdir:
    log_args(args)
  return args


def get_args(atari_defaults=None, mujoco_defaults=None,
             logdir=True, log_period=1, call_log_args=True):
  """ Returns arguments from defaults chosen based on env_id. """
  if atari_defaults is None and mujoco_defaults is None:
    raise ValueError("atari_defaults and mujoco_defaults cannot both be None")
  if call_log_args and not logdir:
    raise ValueError("logdir must be True when call_log_args is True")
  env_type_defaults = dict(atari=atari_defaults, mujoco=mujoco_defaults)
  simple_parser = get_simple_parser(add_logdir=logdir, log_period=log_period)
  args, unknown_args = simple_parser.parse_known_args()

  if is_atari_id(args.env_id):
    env_type = "atari"
  elif is_mujoco_id(args.env_id):
    env_type = "mujoco"
  else:
    defaults_parser = argparse.ArgumentParser()
    choices = set(k for k, v in env_type_defaults.items() if v is not None)
    defaults_parser.add_argument("--defaults", choices=choices)
    args, unknown_args = defaults_parser.parse_known_args(unknown_args, args)
    if args.defaults is None:
      defaults_parser.error(
          f"{args.env_id} is neither an atari nor mujoco env, "
          "please specify which defaults to choose by using "
          f"--defaults {choices}")
    env_type = args.defaults

  defaults = env_type_defaults[env_type]
  if defaults is None:
    raise ValueError(f"cannot run env {args.env_id} because "
                     "{0} defaults are not specified; does this algorithm "
                     "support {0} envs?".format(env_type))

  alg_parser = get_parser(defaults, add_env_id=False, add_logdir=False)
  args = alg_parser.parse_args(unknown_args, args)
  if call_log_args:
    log_args(args)
  return args
