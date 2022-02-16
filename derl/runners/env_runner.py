""" Defines environment runner. """
from abc import ABC, abstractmethod
from collections import defaultdict


class EnvRunner:
  """ Iterable that interacts with an env. """
  def __init__(self, env, policy, horizon, nsteps=None):
    self.env = env
    self.policy = policy
    self.horizon = horizon
    self.nsteps = int(nsteps)
    self.step_count = 0

  @property
  def nenvs(self):
    """ Returns number of batched envs or `None` if env is not batched. """
    return getattr(self.env.unwrapped, "nenvs", None)

  def is_exhausted(self):
    """ Returns `True` if the runner performed predefined number of steps. """
    return self.nsteps is not None and self.step_count >= self.nsteps

  def __len__(self):
    """ Returns the desired number of steps if it was specified, otherwise
    the current step. """
    return self.nsteps if self.nsteps is not None else self.step_count

  def run(self, obs=None):
    """ Interacts with the environment starting from obs for horizon steps. """
    if obs is None:
      obs = self.env.reset()
    while not self.is_exhausted():
      interactions = defaultdict(list)
      for _ in range(self.horizon):
        act = self.policy.act(obs)
        interactions["observations"].append(obs)
        if "actions" not in act:
          raise ValueError("result of policy.act must contain 'actions' "
                           f"but has keys {list(act.keys())}")
        for key, val in act.items():
          interactions[key].append(val)
        new_obs, rew, done, info = self.env.step(act["actions"])
        interactions["rewards"].append(rew)
        interactions["resets"].append(done)
        interactions["infos"].append(info)
        interactions["next_observations"].append(new_obs)

        # Note that batched envs should auto-reset, hence we only check
        # done flag if the env is not batched.
        obs = self.env.reset() if self.nenvs is None and done else new_obs

      interactions["state"] = dict(latest_observations=obs)
      self.step_count += self.horizon * (self.nenvs or 1)
      yield dict(interactions)


class RunnerWrapper(ABC):
  """ Wraps an env runner. """
  def __init__(self, runner):
    self.runner = runner
    self.unwrapped = getattr(runner, "unwrapped", runner)

  def __getattr__(self, attr):
    if attr not in {"env", "policy", "horizon", "nsteps", "step_count",
                    "nenvs", "is_exhausted"}:
      raise AttributeError(f"'{self.__class__.__name__}' "
                           f"has no attribute '{attr}'")
    return getattr(self.runner, attr)

  def __len__(self):
    return len(self.runner)

  @abstractmethod
  def run(self, obs=None):
    """ Interacts with the environment starting from obs for horizon steps. """
