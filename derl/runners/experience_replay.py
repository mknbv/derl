""" Implements experience replay. """
import numpy as np


class InteractionStorage:
  """ Simple circular buffer that stores interactions. """
  def __init__(self, size):
    self.size = size
    self.observations = np.empty(self.size, dtype=np.object)
    self.actions = np.empty(self.size, dtype=np.object)
    self.resets = np.empty(self.size, dtype=np.bool)
    self.rewards = np.empty(self.size, dtype=np.float32)
    self.index = 0
    self.is_full = self.index >= self.size

  def add(self, observation, action, reward, done):
    """ Adds new interaction to the storage. """
    self.observations[self.index] = observation
    self.actions[self.index] = action
    self.rewards[self.index] = reward
    self.resets[self.index] = done
    self.is_full = self.is_full or self.index + 1 == self.size
    self.index = (self.index + 1) % self.size

  def add_batch(self, observations, actions, rewards, resets):
    """ Adds a batch of interactions to the storage. """
    batch_size = observations.shape[0]
    if (batch_size != rewards.shape[0] or batch_size != actions.shape[0]
        or batch_size != resets.shape[0]):
      raise ValueError(
          "observations, actions, rewards, and resets all must have the same "
          "first dimension, got first dim sizes: "
          f"{actions.shape[0]}, {rewards.shape[0]}, {resets.shape[0]}")

    indices = (self.index + np.arange(batch_size)) % self.size
    self.observations[indices] = list(observations)
    self.actions[indices] = list(actions)
    self.rewards[indices] = rewards
    self.resets[indices] = resets
    self.is_full = self.is_full or self.index + batch_size >= self.size
    self.index = (self.index + batch_size) % self.size

  def sample(self, size):
    """ Returns random sample of interactions of specified size. """
    indices = np.random.randint(self.index - 1 if not self.is_full
                                else self.size - 1, size=size)
    nosample_index = (self.index + self.size - 1) % self.size
    inc_mask = indices >= nosample_index
    indices[inc_mask] = (indices[inc_mask] + 1) % self.size

    obs = np.array(list(self.observations[indices]))
    actions = np.array(list(self.actions[indices]))
    rewards = self.rewards[indices]
    resets = self.resets[indices]
    next_indices = (indices + 1) % self.size
    next_obs = np.array(list(self.observations[next_indices]))
    return obs, actions, rewards, resets, next_obs
