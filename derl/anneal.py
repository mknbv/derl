""" Annealing variables. """
from abc import ABC, abstractmethod
import re
import torch
import derl.summary as summary


def camel2snake(string):
  """ Converts string from CamelCase to snake_case. """
  sub = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', string)
  return re.sub('([a-z0-9])([A-Z])', r'\1_\2', sub).lower()


class AnnealingVariable(ABC):
  """ Variable the value of which changes after each step. """
  def __init__(self, name=None):
    self.name = name or camel2snake(self.__class__.__name__)
    self.step_count = 0

  @abstractmethod
  def get_tensor(self):
    """ Returns torch.Tensor that changes after each call to step. """

  def get_current_value(self):
    """ Returns the current value of the variable. """
    return self.get_tensor().clone()

  @abstractmethod
  def step(self):
    """ Update the value of the variable. """

  def step_to(self, val):
    """ Updates the value of the variable `val - self.step_count` times. """
    if val < self.step_count:
      raise ValueError(f"val={val} cannot be smaller than "
                       f"self.step_count={self.step_count}")
    for _ in range(val - self.step_count):
      self.step()

  def summarize(self, global_step):
    """ Writes summary of the value for tensorboard. """
    summary.add_scalar(f"anneal/{self.name}", self.get_tensor(),
                       global_step=global_step)


class TorchSched(AnnealingVariable):
  """ Annealing variable based on torch scheduler. """
  def __init__(self, scheduler, name=None):
    super().__init__(name)
    self.scheduler = scheduler
    # pylint: disable=not-callable
    self.tensor = torch.tensor(self.scheduler.get_last_lr())

  def get_tensor(self):
    return self.tensor

  def step(self):
    self.step_count += 1
    self.scheduler.step()
    # pylint: disable=not-callable
    self.tensor.data = torch.tensor(self.scheduler.get_last_lr())
    return self.get_current_value()


class LinearAnneal(AnnealingVariable):
  """ Linearly annealing variable. """
  def __init__(self, start, nsteps, end=0., name=None):
    super().__init__(name)
    self.start = start
    self.nsteps = nsteps
    self.end = end
    self.tensor = torch.tensor(self.start)  # pylint: disable=not-callable

  def get_tensor(self):
    return self.tensor

  def step(self):
    self.step_count += 1
    step_frac = self.step_count / self.nsteps
    self.tensor.data = torch.clamp(
        # pylint: disable=not-callable
        torch.tensor(self.start + (self.end - self.start) * step_frac),
        min(self.start, self.end),
        max(self.start, self.end)
    )
    return self.get_current_value()
