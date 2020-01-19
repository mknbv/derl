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


def linear_anneal(name, start_value, nsteps, step_var, end_value=0.):
  """ Returns variable that will be linearly annealed. """
  if not isinstance(step_var, StepVariable):
    raise TypeError("step_var must be an instance of StepVariable, "
                    f"got {type(step_var)} instead")

  var = torch.tensor(start_value)  # pylint: disable=not-callable
  step_var.add_annealing_tensor(
      var,
      lambda: torch.clamp(
          torch.tensor(  # pylint: disable=not-callable
              start_value
              + int(step_var) / nsteps * (end_value - start_value)
          ),
          min(start_value, end_value),
          max(start_value, end_value)
      ),
      name)
  return var
