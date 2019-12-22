""" Utils for training. """
import torch
import derl.summary as summary


_GLOBAL_STEP = None


class StepVariable:
  """ Step variable. """
  _global_step = None

  def __init__(self, value=0):
    self.value = value
    self.anneals = []

  @classmethod
  def _check_global_step(cls, should_exist):
    if should_exist and cls._global_step is None:
      raise ValueError("global step does not exist, create it by calling "
                       "create_global_step")
    if not should_exist and cls._global_step is not None:
      raise ValueError(f"global step already exists: {_GLOBAL_STEP}")

  @classmethod
  def create_global_step(cls, value=0):
    """ Creates and returns global step variable. """
    cls._check_global_step(should_exist=False)
    cls._global_step = StepVariable(value)
    return cls._global_step

  @classmethod
  def get_global_step(cls):
    """ Returns global step variable. """
    cls._check_global_step(True)
    return cls._global_step

  @classmethod
  def get_or_create_global_step(cls):
    """ Returns global step which is created if it does already not exist. """
    if cls._global_step is not None:
      return cls._global_step
    return cls.create_global_step()

  @classmethod
  def unset_global_step(cls):
    """ Removes global step variable. """
    step = cls._global_step
    cls._global_step = None
    return step

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
