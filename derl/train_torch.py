""" Utils for training. """
import torch
import derl.summary as summary


_GLOBAL_STEP = None


def create_global_step(value=0):
  """ Creates and returns global step variable. """
  global _GLOBAL_STEP  # pylint: disable=global-statement
  if _GLOBAL_STEP is not None:
    raise ValueError(f"global step already exists: {_GLOBAL_STEP}")
  _GLOBAL_STEP = StepVariable(value)
  return _GLOBAL_STEP


def get_global_step():
  """ Returns global step variable. """
  global _GLOBAL_STEP  # pylint: disable=global-statement
  if _GLOBAL_STEP is None:
    raise ValueError("global step does not exist, create it by calling "
                     "create_global_step")
  return _GLOBAL_STEP


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
      lambda: torch.tensor(  # pylint: disable=not-callable
          start_value + int(step_var) / nsteps * (end_value - start_value)),
      name)
  return var
