""" Defines algorithm base class and various utils. """
from abc import ABC, abstractmethod

import torch
from derl.anneal import camel2snake
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


class Alg(ABC):
  """ Generic learning algorithm specified by its loss function. """
  def __init__(self, runner, optimizer, anneals=None,
               max_grad_norm=None, name=None):
    self.runner = runner
    self.model = self.runner.policy.model
    self.optimizer = optimizer
    self.anneals = anneals if anneals is not None else []
    self.max_grad_norm = max_grad_norm
    if name is None:
      name = camel2snake(self.__class__.__name__)
    self.name = name
    self.step_var = 0

  @abstractmethod
  def loss(self, data):
    """ Computes and returns the loss function on data. """

  def preprocess_gradients(self, parameters):
    """ Applies gradient preprocessing. """
    grad_norm = None
    if self.max_grad_norm is not None:
      grad_norm = torch.nn.utils.clip_grad_norm_(
          parameters, self.max_grad_norm)
    if summary.should_record():
      if grad_norm is None:
        grad_norm = total_norm(p.grad for p in parameters if p.grad is not None)
      summary.add_scalar(f"{self.name}/grad_norm", grad_norm,
                         global_step=self.step_var)

  def step(self, data):
    """ Performs learning step of the algorithm. """
    loss = self.loss(data)
    loss.backward()
    self.preprocess_gradients(self.model.parameters())
    self.optimizer.step()
    self.optimizer.zero_grad()
    for anneal in self.anneals:
      if hasattr(anneal, "summarize") and summary.should_record():
        anneal.summarize(self.step)
      anneal.step()
    self.step_var += 1
    return loss

  def learn(self, obs=None):
    """ Performs learning with this algorithm. """
    for data in self.runner.run(obs=obs):
      yield data, self.step(data)
