""" Implements experience replay. """
import numpy as np
from derl.runners.env_runner import RunnerWrapper
from derl.runners.onpolicy import EnvRunner, TransformInteractions


class InteractionStorage:
  """ Simple circular buffer that stores interactions. """
  def __init__(self, capacity):
    self.capacity = capacity
    self.observations = np.empty(self.capacity, dtype=np.object)
    self.actions = np.empty(self.capacity, dtype=np.object)
    self.resets = np.empty(self.capacity, dtype=np.bool)
    self.rewards = np.empty(self.capacity, dtype=np.float32)
    self.index = 0
    self.is_full = self.index >= self.capacity

  @classmethod
  def from_env(cls, env, capacity, init_size=50_000):
    """ Creates storage and initializes it with random interactions. """
    storage = cls(capacity)
    obs = env.reset()
    for _ in range(init_size):
      action = env.action_space.sample()
      next_obs, rew, done, _ = env.step(action)
      storage.add(obs, action, rew, done)
      obs = next_obs if not done else env.reset()
    return storage

  @property
  def size(self):
    """ Returns the number elements stored. """
    return self.capacity if self.is_full else self.index

  def get(self, indices, nstep=3):
    """ Returns `nstep` interactions starting from indices `indices`. """
    nstep_indices = (indices[:, None] + np.arange(nstep)[None]) % self.capacity
    next_indices = (indices + nstep) % self.capacity
    return {
        "observations": np.array(list(self.observations[indices])),
        "actions": np.array(list(self.actions[indices])),
        "rewards": self.rewards[nstep_indices],
        "resets": self.resets[nstep_indices],
        "next_observations": np.array(list(self.observations[next_indices])),
    }

  def add(self, observation, action, reward, done):
    """ Adds new interaction to the storage. """
    index = self.index
    self.observations[index] = observation
    self.actions[index] = action
    self.rewards[index] = reward
    self.resets[index] = done
    self.is_full = self.is_full or index + 1 == self.capacity
    self.index = (index + 1) % self.capacity
    return index

  def add_batch(self, observations, actions, rewards, resets):
    """ Adds a batch of interactions to the storage. """
    batch_size = observations.shape[0]
    if (batch_size != rewards.shape[0] or batch_size != actions.shape[0]
        or batch_size != resets.shape[0]):
      raise ValueError(
          "observations, actions, rewards, and resets all must have the same "
          "first dimension, got first dim sizes: "
          f"{actions.shape[0]}, {rewards.shape[0]}, {resets.shape[0]}")

    indices = (self.index + np.arange(batch_size)) % self.capacity
    self.observations[indices] = list(observations)
    self.actions[indices] = list(actions)
    self.rewards[indices] = rewards
    self.resets[indices] = resets
    self.is_full = self.is_full or self.index + batch_size >= self.capacity
    self.index = (self.index + batch_size) % self.capacity
    return indices

  def sample(self, size, nstep=3):
    """ Returns random sample of interactions of specified size. """
    indices = np.random.randint(self.index - nstep if not self.is_full
                                else self.capacity - nstep, size=size)
    nosample_index = (self.index + self.capacity - nstep) % self.capacity
    inc_mask = indices >= nosample_index
    indices[inc_mask] = (indices[inc_mask] + nstep) % self.capacity
    return self.get(indices, nstep)


class ExperienceReplay(RunnerWrapper):
  """ Saves interactions to experience replay and samples from it. """
  def __init__(self, runner, storage, batch_size, nstep=3):
    super().__init__(runner)
    self.storage = storage
    self.batch_size = batch_size
    self.nstep = nstep

  def __iter__(self):
    for interactions in self.runner:
      interactions = [interactions[k] for k in ("observations", "actions",
                                                "rewards", "resets")]
      self.storage.add_batch(*interactions)
      yield self.storage.sample(self.batch_size, self.nstep)

# pylint: disable=too-many-arguments
def make_dqn_runner(env, policy, storage_size,
                    steps_per_sample=4,
                    batch_size=32,
                    nstep=3,
                    init_size=50_000,
                    step_var=None):
  """ Creates experience replay runner as used typically used with DQN alg. """
  runner = EnvRunner(env, policy, nsteps=steps_per_sample, step_var=step_var)
  runner = TransformInteractions(runner)
  storage = InteractionStorage.from_env(env, storage_size, init_size)
  return ExperienceReplay(runner, storage, batch_size, nstep)
