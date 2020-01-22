""" Annealing variables. """
from abc import ABC, abstractmethod
import torch


class AnnealingVariable(ABC):
  """ Variable the value of which changes after each step. """
  @abstractmethod
  def get_value(self):
    """ Returns the current value of the variable. """

  @abstractmethod
  def step(self):
    """ Update the value of the variable. """

class TorchSched(AnnealingVariable):
  """ Annealing variable based on torch scheduler. """
  def __init__(self, scheduler):
    self.scheduler = scheduler

  def get_value(self):
    # pylint: disable=not-callable
    return torch.tensor(self.scheduler.get_last_lr())

  def step(self):
    self.scheduler.step()
    return self.get_value()


class LinearAnneal(AnnealingVariable):
  """ Linearly annealing variable. """
  def __init__(self, start, end, num_steps):
    self.start = start
    self.end = end
    self.step_count = 0
    self.num_steps = num_steps
    self.value = torch.tensor(self.start)  # pylint: disable=not-callable

  def get_value(self):
    return self.value

  def step(self):
    self.step_count += 1
    step_frac = self.step_count / self.num_steps
    self.value = torch.clamp(
        # pylint: disable=not-callable
        torch.tensor(self.start + (self.end - self.start) * step_frac),
        min(self.start, self.end),
        max(self.start, self.end)
    )
    return self.value
