""" Sum tree data structure for prioritized experience replay. """
import numpy as np


class SumTree:
  """ Binary tree where leafs are stored elements and sub-tree roots
  contain sum of their leafs. """
  def __init__(self, size):
    self.size = size
    self.data = np.zeros(2 * size - 1)

  @property
  def sum(self):
    """ Sum of all stored elements. """
    return self.data[0]

  def get_value(self, index):
    """ Returns elements value under index. """
    return self.data[index + self.size - 1]

  def replace(self, index, value):
    """ Replaces element under index with new value. """
    index, value = np.array(index), np.array(value)
    if index.shape != value.shape:
      raise ValueError("index and value cannot have different shapes: "
                       f"index.shape={index.shape}, value.shape={value.shape}")
    # pylint: disable=misplaced-comparison-constant
    if index.size and not np.all((0 <= index) & (index < self.size)):
      raise ValueError(f"index out of bounds [0, {self.size}): {index}")
    if np.unique(index).size != index.size:
      raise ValueError(f"index must be unique, got {index}")
    index += self.size - 1
    old_value = np.asarray(self.data[index])
    self.data[index] = value
    while index.size:
      mask = index.astype(np.bool)
      index = index[mask] - 1 >> 1
      value, old_value = value[mask], old_value[mask]
      np.add.at(self.data, index, value - old_value)

  def retrieve(self, value):
    """ Returns element under index `i` such that `sum(elements[:i])`
    is closest to `value` without being less than it. """
    value = np.array(value)
    index = np.zeros_like(value, dtype=np.int32)
    while np.min(index) < self.size - 1:
      mask = index < self.size - 1
      masked_index, masked_value = index[mask], value[mask]
      left, right = 2 * masked_index + 1, 2 * masked_index + 2
      go_left = masked_value <= self.data[left]
      go_right = ~go_left

      masked_index[go_left] = left[go_left]
      masked_index[go_right] = right[go_right]
      index[mask] = masked_index
      masked_value[go_right] -= self.data[left[go_right]]
      value[mask] = masked_value
    return index - self.size + 1
