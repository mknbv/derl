"""
Defines base classes.
"""
from abc import ABC, abstractmethod

import torch
import derl.summary_manager as summary_manager
from derl.train_torch import StepVariable


class BaseAlgorithm(ABC):
  """ Base algorithm. """
  def __init__(self, model, optimizer, step_var=None):
    self.model = model
    self.optimizer = optimizer
    if step_var is None:
      step_var = StepVariable()
    self.step_var = step_var

  @abstractmethod
  def loss(self, data):
    """ Computes the loss given inputs and target values. """

  def preprocess_gradients(self, parameters):
    """ Applies gradient preprocessing. """
    if hasattr(self, "max_grad_norm"):
      grad_norm = torch.nn.utils.clip_grad_norm_(
          parameters, getattr(self, "max_grad_norm"))
      if summary_manager.should_record_summaries():
        summary_manager.add_scalar(
            f"{self.__class__.__name__.lower()}/grad_norm", grad_norm)

  def step(self, data):
    """ Performs single training step of the algorithm. """
    loss = self.loss(data)
    loss.backward()
    self.preprocess_gradients(self.model.parameters())
    self.optimizer.step()
    self.optimizer.zero_grad()
    self.step_var.assign_add(1)
    return loss
