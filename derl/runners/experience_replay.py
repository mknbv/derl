""" Implements experience replay. """
from functools import partial
import numpy as np
from derl.runners.env_runner import EnvRunner, RunnerWrapper
from derl.runners.onpolicy import TransformInteractions
from derl.runners.storage import InteractionStorage, PrioritizedStorage
from derl.train import linear_anneal


class ExperienceReplay(RunnerWrapper):
  """ Saves interactions to storage and samples from it. """
  def __init__(self, runner, storage, storage_init_size=50_000,
               batch_size=32, nstep=3):
    super().__init__(runner)
    self.storage = storage
    self.storage_init_size = storage_init_size
    self.initialized_storage = False
    self.batch_size = batch_size
    self.nstep = nstep

  def initialize_storage(self, obs=None):
    """ Initializes the storage with random interactions with environment. """
    if self.initialized_storage:
      raise ValueError("storage is already initialized")
    if self.storage.size != 0:
      raise ValueError(f"storage has size {self.storage.size}, but "
                       "but initialization requires it to be empty")
    if obs is None:
      obs = self.env.reset()
    for _ in range(self.storage_init_size):
      action = self.env.action_space.sample()
      next_obs, rew, done, _ = self.env.step(action)
      self.storage.add(obs, action, rew, done)
      obs = next_obs if not done else self.env.reset()
    self.initialized_storage = True
    return obs

  def run(self, obs=None):
    if not self.initialized_storage:
      obs = self.initialize_storage(obs=obs)
    for interactions in self.runner.run(obs=obs):
      interactions = [interactions[k] for k in ("observations", "actions",
                                                "rewards", "resets")]
      self.storage.add_batch(*interactions)
      yield self.storage.sample(self.batch_size, self.nstep)


class PrioritizedExperienceReplay(ExperienceReplay):
  """ Experience replay with prioritized storage. """
  def __init__(self, runner, storage, alpha=0.6, beta=(0.4, 1),
               epsilon=1e-8, **experience_replay_kwargs):
    super().__init__(runner, storage, **experience_replay_kwargs)
    if not hasattr(storage, "update_priorities"):
      raise ValueError("storage does not implement `update_priorities` "
                       "method")
    if isinstance(beta, (tuple, list)):
      if len(beta) != 2:
        raise ValueError("beta must be a float, a tuple or a list of length 2 "
                         f"got len(beta)={len(beta)}")
      if self.runner.nsteps is None:
        raise ValueError("when beta is a tuple of (start, end) values "
                         "but runner.nsteps cannot be None")
      beta = linear_anneal("per_beta", beta[0], self.runner.nsteps,
                           self.runner.step_var, beta[1])
    self.alpha = alpha
    self.beta = beta
    self.epsilon = epsilon

  def update_priorities(self, errors, indices):
    """ Updates priorities for specified inidices. """
    # Need to as well update priorities for interactions that occurred before
    # those, for which errors are computed as in the paper.
    mask = ~self.storage.get(indices, nstep=1)["resets"][:, 0]
    if not self.storage.is_full:
      mask &= indices > 0
    capacity = self.storage.capacity
    prev_indices = (indices[mask] - 1 + capacity) % capacity

    indices = np.concatenate([prev_indices, indices], 0)
    errors = np.concatenate([errors[mask] + self.epsilon, errors], 0)
    priorities = np.power(errors, self.alpha)
    self.storage.update_priorities(indices, priorities)

  def run(self, obs=None):
    for interactions in super().run(obs=obs):
      if not isinstance(self.beta, (float, int)):
        beta = float(self.beta.numpy())
      log_weights = -beta * (
          np.log(self.storage.size) + interactions["log_probs"])
      interactions["weights"] = np.exp(log_weights - np.max(log_weights))
      interactions["update_priorities"] = partial(
          self.update_priorities, indices=interactions["indices"])
      yield interactions


def dqn_runner_wrap(runner, prioritized=True,
                    storage_size=1_000_000, storage_init_size=50_000,
                    batch_size=32, nstep=3, **prioritized_kwargs):
  """ Wraps runner as it is typically used with DQN alg. """
  if prioritized:
    storage = PrioritizedStorage(storage_size)
    return PrioritizedExperienceReplay(
        runner, storage, **prioritized_kwargs,
        storage_init_size=storage_init_size,
        batch_size=batch_size, nstep=nstep)
  storage = InteractionStorage(storage_size)
  return ExperienceReplay(runner, storage, storage_init_size=storage_init_size,
                          batch_size=batch_size, nstep=nstep)

def make_dqn_runner(env, policy, num_train_steps, steps_per_sample=4,
                    step_var=None, **wrap_kwargs):
  """ Creates experience replay runner as used typically used with DQN alg. """
  runner = EnvRunner(env, policy, horizon=steps_per_sample,
                     nsteps=num_train_steps, step_var=step_var)
  runner = TransformInteractions(runner)
  return dqn_runner_wrap(runner, **wrap_kwargs)
