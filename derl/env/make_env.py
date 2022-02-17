""" Creates environments with standard wrappers. """
import gym
try:
  # Populates gym registry with pybullet envs
  import pybullet_envs   # pylint: disable=unused-import
except ImportError:
  pass  # pylint: disable=bare-except
from atari_py import list_games
from .atari_wrappers import (
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
from .mujoco_wrappers import Normalize, TanhRangeActions
from .summarize import Summarize


def list_envs(env_type):
  """ Returns list of envs ids of given type. """
  ids = {
      "atari": list("".join(c.capitalize() for c in g.split("_"))
                    for g in list_games()),
      "mujoco": [
          "Reacher",
          "Pusher",
          "Thrower",
          "Striker",
          "InvertedPendulum",
          "InvertedDoublePendulum",
          "HalfCheetah",
          "Hopper",
          "Swimmer",
          "Walker2d",
          "Ant",
          "Humanoid",
          "HumanoidStandup",
      ]
  }
  return ids[env_type]


def is_atari_id(env_id):
  """ Returns True if env_id corresponds to an Atari env. """
  env_id = env_id[:env_id.rfind("-")]
  for postfix in ("Deterministic", "NoFrameskip"):
    if env_id.endswith(postfix):
      env_id = env_id[:-len(postfix)]

  atari_envs = set(list_envs("atari"))
  return env_id in atari_envs


def is_mujoco_id(env_id):
  """ Returns True if env_id corresponds to MuJoCo env. """
  env_id = "".join(env_id.split("-")[:-1])
  if env_id.endswith("BulletEnv"):
    env_id = env_id[:-len("BulletEnv")]
  mujoco_ids = set(list_envs("mujoco"))
  return env_id in mujoco_ids


def get_seed(nenvs=None, seed=None):
  """ Returns seed(s) for specified number of envs. """
  if nenvs is None and seed is not None and not isinstance(seed, int):
    raise ValueError("when nenvs is None seed must be None or an int, "
                     f"got type {type(seed)}")
  if nenvs is None:
    return seed or 0
  if isinstance(seed, (list, tuple)):
    if len(seed) != nenvs:
      raise ValueError(f"seed must have length {nenvs} but has {len(seed)}")
    return seed
  if seed is None:
    seed = list(range(nenvs))
  elif isinstance(seed, int):
    seed = [seed] * nenvs
  else:
    raise ValueError(f"invalid seed: {seed}")
  return seed


def set_seed(env, seed=None):
  """ Sets seed of a given env. """
  env.seed(seed)
  env.action_space.np_random.seed(seed)


def nature_dqn_env(env_id, nenvs=None, seed=None,
                   summarize=True, episodic_life=True, clip_reward=True):
  """ Wraps env as in Nature DQN paper. """
  assert is_atari_id(env_id)
  if "NoFrameskip" not in env_id:
    raise ValueError(f"env_id must have 'NoFrameskip' but is {env_id}")
  seed = get_seed(nenvs)
  if nenvs is not None:
    env = ParallelEnvBatch([
        lambda i=i, s=s: nature_dqn_env(
            env_id, seed=s, summarize=False,
            episodic_life=episodic_life, clip_reward=False)
        for i, s in enumerate(seed)
    ])
    if summarize:
      env = Summarize.reward_summarizer(env, prefix=env_id)
    if clip_reward:
      env = ClipReward(env)
    return env

  env = gym.make(env_id)
  set_seed(env, seed)
  return nature_dqn_wrap(env, summarize=summarize,
                         episodic_life=episodic_life,
                         clip_reward=clip_reward)


def nature_dqn_wrap(env, summarize=True, episodic_life=True, clip_reward=True):
  """ Wraps given env as in nature DQN paper. """
  if summarize:
    env = Summarize.reward_summarizer(env)
  if episodic_life:
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


def mujoco_env(env_id, nenvs=None, seed=None, time_limit=True, **kwargs):
  """ Creates and wraps MuJoCo env. """
  assert is_mujoco_id(env_id)
  seed = get_seed(nenvs, seed)
  if nenvs is not None:
    env = ParallelEnvBatch([
        lambda s=s: mujoco_env(env_id, seed=s, time_limit=time_limit,
                               summarize=False, normalize_obs=False,
                               normalize_ret=False, tanh_range_actions=False)
        for s in seed])
    return mujoco_wrap(env, **kwargs)

  env = gym.make(env_id)
  set_seed(env, seed)
  if not time_limit:
    env = env.env
  return mujoco_wrap(env, **kwargs)


def mujoco_wrap(env, summarize=True, normalize_obs=True, normalize_ret=True,
                tanh_range_actions=False):
  """ Wraps given env as a mujoco env. """
  if summarize:
    env = Summarize.reward_summarizer(env)
  if normalize_obs or normalize_ret:
    env = Normalize(env, obs=normalize_obs, ret=normalize_ret)
  if tanh_range_actions:
    env = TanhRangeActions(env)
  return env


def make(env_id, nenvs=None, seed=None, **kwargs):
  """ Creates env with standard wrappers. """
  if is_atari_id(env_id):
    return nature_dqn_env(env_id, nenvs, seed=seed, **kwargs)
  if is_mujoco_id(env_id):
    return mujoco_env(env_id, nenvs, seed=seed, **kwargs)

  def _make(seed):
    env = gym.make(env_id, **kwargs)
    set_seed(env, seed)
    return env

  seed = get_seed(nenvs, seed)
  if nenvs is None:
    return _make(seed)
  return ParallelEnvBatch([lambda s=s: _make(s) for s in seed])
