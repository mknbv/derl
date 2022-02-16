""" Implements experience replay. """
from functools import partial
import numpy as np
from derl.anneal import LinearAnneal
from derl.runners.env_runner import EnvRunner, RunnerWrapper
from derl.runners.onpolicy import TransformInteractions
from derl.runners.storage import InteractionStorage, PrioritizedStorage
from derl.runners.summary import PeriodicSummaries
from derl.runners.trajectory_transforms import NoResets
from derl import summary


class ExperienceReplay(RunnerWrapper):
  """ Saves interactions to storage and samples from it. """
  def __init__(self, runner, storage,
               storage_init_size=50_000,
               batch_size=32,
               store_next_observations=False,
               anneals=None):
    super().__init__(runner)
    self.storage = storage
    self.storage_init_size = storage_init_size
    self.batch_size = batch_size
    self.store_next_observations = store_next_observations
    if anneals is None:
      anneals = []
    self.anneals = tuple(anneals)
    self.initialized_storage = False

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
      self.storage.add(obs, action, rew, done, next_obs)
      obs = next_obs if not done else self.env.reset()
    self.initialized_storage = True
    return obs

  def run(self, obs=None):
    if not self.initialized_storage:
      obs = self.initialize_storage(obs=obs)
    for interactions in self.runner.run(obs=obs):
      interactions = {k: interactions[k] for k in
                      ("observations", "actions", "rewards", "resets",
                       "next_observations")}
      self.storage.add_batch(**interactions)
      for anneal in self.anneals:
        if summary.should_record():
          anneal.summarize(self.step_count)
        anneal.step_to(self.step_count)
      yield self.storage.sample(self.batch_size)


class PrioritizedExperienceReplay(ExperienceReplay):
  """ Experience replay with prioritized storage. """
  def __init__(self, runner, storage,
               alpha=0.6,
               beta=(0.4, 1),
               epsilon=1e-8,
               anneals=None,
               **experience_replay_kwargs):
    if anneals is None:
      anneals = []
    anneals = list(anneals)
    if not hasattr(storage, "update_priorities"):
      raise ValueError("storage does not implement `update_priorities` "
                       "method")
    if isinstance(beta, (tuple, list)):
      if len(beta) != 2:
        raise ValueError("beta must be a float, a tuple or a list of length 2 "
                         f"got len(beta)={len(beta)}")
      if runner.nsteps is None:
        raise ValueError("when beta is a tuple of (start, end) values "
                         "runner.nsteps cannot be None")
      beta_anneal = LinearAnneal(beta[0], runner.nsteps, beta[1], "per_beta")
      beta = beta_anneal.get_tensor()
      anneals.append(beta_anneal)
    super().__init__(runner, storage, anneals=anneals,
                     **experience_replay_kwargs)
    self.alpha = alpha
    self.beta = beta
    self.epsilon = epsilon

  def update_priorities(self, errors, indices):
    """ Updates priorities for specified inidices. """
    # Need to as well update priorities for interactions that occurred before
    # those, for which errors are computed as in the paper.
    mask = ~self.storage.get(indices)["resets"][:, 0]
    if not self.storage.is_full:
      mask &= indices > 0
    capacity = self.storage.capacity
    prev_indices = (indices - 1 + capacity) % capacity
    mask &= ~np.isin(prev_indices, indices)

    indices = np.concatenate([prev_indices[mask], indices], 0)
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


def dqn_runner_wrap(runner, prioritized=True, storage_size=1_000_000,
                    store_next_observations=False, nstep=3, **kwargs):
  """ Wraps runner as it is typically used with DQN alg. """
  if prioritized:
    storage = PrioritizedStorage(storage_size, nstep)
    return PrioritizedExperienceReplay(runner, storage, **kwargs)
  storage = InteractionStorage(storage_size, nstep, store_next_observations)
  return ExperienceReplay(runner, storage, **kwargs)


def make_dqn_runner(env, policy, num_train_steps, steps_per_sample=4,
                    nlogs=1e5, **wrap_kwargs):
  """ Creates experience replay runner as typically used with DQN alg. """
  runner = EnvRunner(env, policy, horizon=steps_per_sample,
                     nsteps=num_train_steps)
  runner = PeriodicSummaries.make_with_nlogs(runner, nlogs)
  runner = TransformInteractions(runner)
  return dqn_runner_wrap(runner, **wrap_kwargs)


class ResampleStorage(RunnerWrapper):
  """ Resamples from specified number of times. """
  def __init__(self, runner, num_samples, batch_size=None):
    if not hasattr(runner, "storage"):
      raise TypeError("runner must have 'storage' attribute, got runner "
                      f"{runner}")
    if batch_size is None:
      if not hasattr(runner, "batch_size"):
        raise TypeError("when batch_size is None runner must have 'batch_size' "
                        f"attribute, got runner {runner}")
      batch_size = runner.batch_size
    super().__init__(runner)
    self.num_samples = num_samples
    self.batch_size = batch_size

  def run(self, obs=None):
    for interactions in self.runner.run(obs=obs):
      yield interactions
      for _ in range(self.num_samples):
        yield self.runner.storage.sample(self.batch_size)


def make_mujoco_sac_runner(env, policy, num_train_steps,
                           batch_size=256,
                           steps_per_sample=1000,
                           num_storage_samples=1000,
                           **kwargs):
  """ Creates SAC runner for mujoco env. """
  runner = make_dqn_runner(env, policy, num_train_steps,
                           batch_size=batch_size,
                           steps_per_sample=steps_per_sample,
                           prioritized=False, nstep=1,
                           store_next_observations=True, **kwargs)
  runner = ResampleStorage(runner, num_storage_samples - 1)
  runner = TransformInteractions(runner, transforms=[NoResets()], asarray=False)
  return runner
