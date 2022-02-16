""" Defines classes that store interactions. """
from collections import deque
from operator import itemgetter
import numpy as np
from derl.runners.sum_tree import SumTree


def maybe_numpy(data):
  """ Tries to cast data to np.ndarray. """
  if isinstance(data, (list, tuple)) and not data:
    return np.array(data)
  return np.array(list(data)) if isinstance(data[0], np.ndarray) else data


class InteractionArrays:
  """ Stores arrays of interactions. """
  def __init__(self, size):
    self.size = size
    self.observations = np.empty(self.size, dtype=object)
    self.actions = np.empty(self.size, dtype=object)
    self.rewards = np.empty(self.size, dtype=np.float32)
    self.resets = np.empty(self.size, dtype=bool)

  def get(self, indices, nstep, next_observations=True):
    """ Returns `nstep` interactions starting from indices `indices`. """
    # pylint: disable=misplaced-comparison-constant
    nstep_indices = (
        (indices[:, None] + np.arange(nstep)[None]) % self.size)
    next_indices = (indices + nstep) % self.size
    result = {
        "observations": maybe_numpy(self.observations[indices]),
        "actions": maybe_numpy(self.actions[indices]),
        "rewards": self.rewards[nstep_indices],
        "resets": self.resets[nstep_indices],
        "next_observations": self.observations[next_indices],
    }
    if next_observations:
      next_indices = (indices + nstep) % self.size
      result["next_observations"] = maybe_numpy(self.observations[next_indices])
    return result

  def set(self, indices, observations, actions, rewards, resets):
    """ Sets values under specified indices. """
    self.observations[indices] = list(observations)
    self.actions[indices] = list(actions)
    self.rewards[indices] = rewards
    self.resets[indices] = resets


class InteractionStorage:
  """ Simple circular buffer that stores interactions. """
  def __init__(self, capacity, nstep=3, store_next_observations=False):
    self.capacity = capacity
    self.nstep = nstep
    if store_next_observations and nstep != 1:
      raise ValueError("when store_next_observations is True, "
                       f"nstep must be 1, got nstep={nstep}")
    self.store_next_observations = store_next_observations
    self.arrays = InteractionArrays(self.capacity)
    self.index = 0
    self.is_full = self.index >= self.capacity

  @property
  def size(self):
    """ Returns the number elements stored. """
    return self.capacity if self.is_full else self.index

  def get(self, indices):
    """ Returns `nstep` interactions starting from indices `indices`. """
    # pylint: disable=misplaced-comparison-constant
    if indices.size and not np.all((0 <= indices) & (indices < self.size)):
      raise ValueError(f"indices out of range(0, {self.size}): {indices}")
    result = self.arrays.get(indices, self.nstep,
                             next_observations=not self.store_next_observations)
    if self.store_next_observations:
      result["observations"], result["next_observations"] = map(
          maybe_numpy, (list(map(itemgetter(0), result["observations"])),
                        list(map(itemgetter(1), result["observations"])))
      )
    return result

  def check_next_observations(self, next_observations):
    """ Checks if next_observations is specified when needed. """
    if self.store_next_observations and next_observations is None:
      raise ValueError("when store_next_observations was set to True "
                       "next_observation cannot be None")

  def add(self, observation, action, reward, done, next_observation=None):
    """ Adds new interaction to the storage. """
    self.check_next_observations(next_observation)
    index = self.index
    if self.store_next_observations:
      observation = observation, next_observation
    self.arrays.set([index], [observation], [action], [reward], [done])
    self.is_full = self.is_full or index + 1 == self.capacity
    self.index = (index + 1) % self.capacity
    return index

  def add_batch(self, observations, actions, rewards, resets,
                next_observations=None):
    """ Adds a batch of interactions to the storage. """
    self.check_next_observations(next_observations)
    batch_size = len(observations)
    if (batch_size != len(rewards) or batch_size != len(actions)
        or batch_size != len(resets)):
      raise ValueError(
          "observations, actions, rewards, and resets all must have the same "
          f"first dimension, got first dim sizes: {observations.shape[0]}, "
          f"{actions.shape[0]}, {rewards.shape[0]}, {resets.shape[0]}")

    if self.store_next_observations:
      observations = list(zip(observations, next_observations))
    indices = (self.index + np.arange(batch_size)) % self.capacity
    self.arrays.set(indices, observations, actions, rewards, resets)
    self.is_full = self.is_full or self.index + batch_size >= self.capacity
    self.index = (self.index + batch_size) % self.capacity
    return indices

  def sample(self, size):
    """ Returns random sample of interactions of specified size. """
    if self.store_next_observations:
      indices = np.random.randint(self.index if not self.is_full
                                  else self.capacity, size=size)
      return self.get(indices)

    # Always sample by nstep less indices than available first,
    # then rearrange the indices such that 'next_observations' is
    # taken correctly.
    indices = np.random.randint(self.index - self.nstep if not self.is_full
                                else self.capacity - self.nstep, size=size)
    # Indices at most nstep before self.index should not be sampled.
    nosample_index = (self.index + self.capacity - self.nstep) % self.capacity
    # If no cycle move occured, then increment all sampled indices
    # above nosample_index by nstep. Note that when sampling
    # indices the capacity excludes nstep last steps so there is no
    # overflow.
    if nosample_index < self.index:
      inc_mask = nosample_index <= indices
      indices[inc_mask] = indices[inc_mask] + self.nstep
    elif self.index:
      # Otherwise, a cycle move occured and we need to increment
      # all indices by self.index
      indices += self.index
    return self.get(indices)


class PrioritizedStorage(InteractionStorage):
  """ Wraps given storage to make it prioritized. """
  def __init__(self, capacity, nstep=3, start_max_priority=1):
    super().__init__(capacity, nstep, store_next_observations=False)
    self.sum_tree = SumTree(capacity)
    self.start_max_priority = start_max_priority
    self.num_pending = self.nstep
    self.pending_indices = deque([])

  def add(self, observation, action, reward, done, next_observation=None):
    """ Adds data to storage. """
    _ = next_observation
    index = None
    if len(self.pending_indices) == self.num_pending:
      index = self.pending_indices.popleft()
    newindex = super().add(observation, action, reward, done)
    self.pending_indices.append(newindex)

    indices, priorities = [newindex], [0]
    if index is not None:
      indices.append(index)
      priorities.append(self.start_max_priority)
    self.sum_tree.replace(indices, priorities)
    return index

  def add_batch(self, observations, actions, rewards, resets,
                next_observations=None):
    """ Adds batch of data to storage. """
    _ = next_observations
    newindices = super().add_batch(observations, actions, rewards, resets)
    self.pending_indices.extend(newindices)
    n = len(self.pending_indices) - self.num_pending
    indices = np.array([self.pending_indices.popleft() for _ in range(n)])
    num_new_pending = min(newindices.size, len(self.pending_indices))
    newindices = newindices[-num_new_pending:]
    priorities = np.concatenate([np.full(indices.size, self.start_max_priority),
                                 np.zeros(newindices.size)], 0)
    indices = np.concatenate([indices, newindices], 0)
    self.sum_tree.replace(indices, priorities)
    return indices

  def sample(self, size):
    """ Samples data from storage. """
    sums = np.linspace(0, self.sum_tree.sum, size + 1)
    samples = np.random.uniform(sums[:-1], sums[1:])
    # In some rare cases it might happen that sampling for different
    # sums leads to same indices, but updating priorities works only with
    # unique indices, which is why we only return unique indices here.
    indices = np.unique(self.sum_tree.retrieve(samples))
    sample = super().get(indices)
    sample["indices"] = indices
    sample["log_probs"] = (np.log(self.sum_tree.get_value(indices))
                           - np.log(self.sum_tree.sum))
    return sample

  def update_priorities(self, indices, priorities):
    """ Updates priorities. """
    self.sum_tree.replace(indices, priorities)
