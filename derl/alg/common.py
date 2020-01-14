""" Defines algorithm base class and various utils. """
from abc import ABC, abstractmethod

import torch
from derl.train import StepVariable
import derl.summary as summary


def r_squared(targets, predictions):
  """ Computes coefficient of determination. """
  variance = torch.pow(predictions.std(), 2)
  return 1. - torch.mean(torch.pow(predictions - targets, 2)) / variance


def torch_from_numpy(arr, device=None):
  """ Creates torch.Tensor from numpy array and collocates it with device. """
  return torch.from_numpy(arr).to(device=device)


def total_norm(tensors, norm_type=2):
  """ Computes total norm of the tensors as if concatenated into 1 vector. """
  if norm_type == float('inf'):
    return max(t.abs().max() for t in tensors)
  return sum(t.norm(norm_type) ** norm_type
             for t in tensors) ** (1. / norm_type)


class BaseAlgorithm(ABC):
  """ Base algorithm. """
  def __init__(self, model, optimizer, step_var=None):
    self.model = model
    self.optimizer = optimizer
    if step_var is None:
      step_var = StepVariable()
    self.step_var = step_var

  @property
  def device(self):
    """ Returns device of the inner model. """
    return next(self.model.parameters()).device

  @abstractmethod
  def loss(self, data):
    """ Computes the loss given inputs and target values. """

  def preprocess_gradients(self, parameters):
    """ Applies gradient preprocessing. """
    grad_norm = None
    if hasattr(self, "max_grad_norm"):
      grad_norm = torch.nn.utils.clip_grad_norm_(
          parameters, getattr(self, "max_grad_norm"))
    if summary.should_record():
      if grad_norm is None:
        grad_norm = total_norm(p.grad for p in parameters if p.grad is not None)
      summary.add_scalar(f"{self.__class__.__name__.lower()}/grad_norm",
                         grad_norm, global_step=self.step_var)

  def step(self, data):
    """ Performs single training step of the algorithm. """
    loss = self.loss(data)
    loss.backward()
    self.preprocess_gradients(self.model.parameters())
    self.optimizer.step()
    self.optimizer.zero_grad()
    self.step_var.assign_add(1)
    return loss
