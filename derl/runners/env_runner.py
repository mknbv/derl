""" Defines environment runner. """
from collections import defaultdict
import tensorflow as tf
from derl.train import StepVariable


class EnvRunner:
  """ Iterable that interacts with an env. """
  def __init__(self, env, policy, batch_size, step_var=None, nsteps=None):
    self.env = env
    self.policy = policy
    if step_var is None:
      step_var = StepVariable("env_runner_step", tf.train.create_global_step())
    self.step_var = step_var
    self.nsteps = nsteps
    self.batch_size = batch_size

  @property
  def nenvs(self):
    """ Returns number of batched envs or `None` if env is not batched. """
    return getattr(self.env.unwrapped, "nenvs", None)

  def is_exhausted(self):
    """ Returns `True` if the runner performed predefined number of steps. """
    return self.nsteps is not None and int(self.step_var) >= self.nsteps

  def __len__(self):
    """ Returns the desired number of steps if it was specified, otherwise
    the current step. """
    return int(self.nsteps if self.nsteps is not None else self.step_var)

  def __iter__(self):
    obs = self.env.reset()
    while not self.is_exhausted():
      interactions = defaultdict(list)
      for _ in range(self.batch_size):
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

        # Note that batched envs should auto-reset, hence we only check
        # done flag if the env is not batched.
        obs = self.env.reset() if self.nenvs is None and done else new_obs

      interactions["state"] = dict(latest_observations=obs)
      self.step_var.assign_add(self.batch_size * (self.nenvs or 1))
      yield dict(interactions)
