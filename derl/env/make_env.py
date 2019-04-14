""" Creates environments with standard wrappers. """
import gym
from .atari_wrappers import (
    Summaries,
    EpisodicLife,
    FireReset,
    StartWithRandomActions,
    MaxBetweenFrames,
    SkipFrames,
    ImagePreprocessing,
    QueueFrames,
    ClipReward,
)
from .env_batch import ParallelEnvBatch
from .mujoco_wrappers import Normalize


def is_atari_id(env_id):
  """ Returns True if env_id corresponds to an Atari env. """
  env_id = "".join(env_id.split("-")[:-1])
  for postfix in ("Deterministic", "NoFrameskip"):
    if env_id.endswith(postfix):
      env_id = env_id[:-len(postfix)]

  # TODO: rewrite without atari_py dependency.
  from atari_py import list_games
  games = set("".join(c.capitalize() for c in g.split("_"))
              for g in list_games())
  return env_id in games


def is_mujoco_id(env_id):
  """ Returns True if env_id corresponds to MuJoCo env. """
  mujoco_ids = {
      "Reacher",
      "Pusher",
      "Thrower",
      "Striker",
      "InvertedPendulum",
      "InvertedDoublePendulum",
      "HalfCheetah",
      "Hopper",
      "Swimmer",
      "Walker",
      "Ant",
      "Humanoid",
      "HumanoidStandup",
  }
  env_id = "".join(env_id.split("-")[:-1])
  return env_id in mujoco_ids


def get_seed(nenvs=None, seed=None):
  """ Returns seed(s) for specified number of envs. """
  if nenvs is None and seed is not None and not isinstance(seed, int):
    raise ValueError("when nenvs is None seed must be None or an int, "
                     f"got type {type(seed)}")
  if nenvs is None:
    return seed or 0
  if seed is not None and len(seed) != nenvs:
    raise ValueError(f"seed must have length {nenvs} but has {len(seed)}")
  if seed is None:
    seed = list(range(nenvs))
  elif isinstance(seed, int):
    seed = [seed] * nenvs
  return seed


def nature_dqn_env(env_id, nenvs=None, seed=None,
                   summaries=True, clip_reward=True):
  """ Wraps env as in Nature DQN paper. """
  assert is_atari_id(env_id)
  if "NoFrameskip" not in env_id:
    raise ValueError(f"env_id must have 'NoFrameskip' but is {env_id}")
  seed = get_seed(nenvs)
  if nenvs is not None:
    env = ParallelEnvBatch([
        lambda i=i, s=s: nature_dqn_env(
            env_id, seed=s, summaries=False, clip_reward=False)
        for i, s in enumerate(seed)
    ])
    if summaries:
      env = Summaries(env, prefix=env_id)
    if clip_reward:
      env = ClipReward(env)
    return env

  env = gym.make(env_id)
  env.seed(seed)
  if summaries:
    env = Summaries(env)
  env = EpisodicLife(env)
  if "FIRE" in env.unwrapped.get_action_meanings():
    env = FireReset(env)
  env = StartWithRandomActions(env, max_random_actions=30)
  env = MaxBetweenFrames(env)
  env = SkipFrames(env, 4)
  env = ImagePreprocessing(env, width=84, height=84, grayscale=True)
  env = QueueFrames(env, 4)
  if clip_reward:
    env = ClipReward(env)
  return env


def mujoco_env(env_id, nenvs=None, seed=None, summaries=True, normalize=True):
  """ Creates and wraps MuJoCo env. """
  assert is_mujoco_id(env_id)
  seed = get_seed(nenvs, seed)
  if nenvs is not None:
    env = ParallelEnvBatch([
        lambda s=s: mujoco_env(env_id, seed=s, summaries=False, normalize=False)
        for s in seed])
    if summaries:
      env = Summaries(env)
    if normalize:
      env = Normalize(env)
    return env

  env = gym.make(env_id)
  env.seed(seed)
  if summaries:
    env = Summaries(env)
  if normalize:
    env = Normalize(env)
  return env


def make(env_id, nenvs=None, seed=None):
  """ Creates an atari or mujoco env with standard wrappers. """
  if is_atari_id(env_id):
    return nature_dqn_env(env_id, nenvs, seed=seed)
  if is_mujoco_id(env_id):
    return mujoco_env(env_id, nenvs, seed=seed)
  if nenvs is None:
    return gym.make(env_id)

  def _make(seed):
    env = gym.make(env_id)
    env.seed(seed)
    return env
  seed = get_seed(nenvs, seed)
  return ParallelEnvBatch([lambda s=s: _make(s) for s in seed])
