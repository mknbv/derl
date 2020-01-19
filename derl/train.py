""" Utils for training. """
import torch
import derl.summary as summary


class StepVariable:
  """ Step variable. """
  def __init__(self, value=0):
    self.value = value
    self.anneals = []

  def __int__(self):
    return self.value

  def assign_add(self, value):
    """ Updates the step variable by incrementing it by value. """
    self.value += value
    for var, fun, name in self.anneals:
      var.data = fun()
      if name is not None and summary.should_record():
        summary.add_scalar(f"train/{name}", var, global_step=int(self))

  def add_annealing_tensor(self, tensor, function, name=None):
    """ Adds annealing tensor. """
    self.anneals.append((tensor, function, name))

  def linear_anneal(self, start_value, nsteps, end_value=0., name=None):
    """ Returns tensor which will be annealed with each update of the step. """
    var = torch.tensor(start_value)  # pylint: disable=not-callable
    self.add_annealing_tensor(
        var,
        lambda: torch.clamp(
            torch.tensor(  # pylint: disable=not-callable
                start_value
                + int(self) / nsteps * (end_value - start_value)
            ),
            min(start_value, end_value),
            max(start_value, end_value)
        ),
        name)
    return var
