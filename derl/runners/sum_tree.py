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

  def add(self, index, value):
    """ Adds value to element under index. """
    index += self.size - 1
    self.data[index] += value
    while index:
      index = index - 1 >> 1
      self.data[index] += value

  def replace(self, index, value):
    """ Replaces element under index with new value. """
    index += self.size - 1
    old_value = self.data[index]
    self.data[index] = value
    while index:
      index = index - 1 >> 1
      self.data[index] = self.data[index] - old_value + value

  def retrieve(self, value):
    """ Returns element under index `i` such that `sum(elements[:i])`
    is closest to `value` without being less than it. """
    index = 0
    while index < self.size - 1:
      left, right = 2 * index + 1, 2 * index + 2
      if value <= self.data[left]:
        index = left
      else:
        value -= self.data[left]
        index = right
    return index - self.size + 1
