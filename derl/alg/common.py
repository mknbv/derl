""" Defines algorithm base class and various utils. """
from abc import ABC, abstractmethod

import torch
import derl.summary as summary


def r_squared(targets, predictions):
  """ Computes coefficient of determination. """
  variance = torch.pow(predictions.std(), 2)
  return 1. - torch.mean(torch.pow(predictions - targets, 2)) / variance


def total_norm(tensors, norm_type=2):
  """ Computes total norm of the tensors as if concatenated into 1 vector. """
  if norm_type == float('inf'):
    return max(t.abs().max() for t in tensors)
  return sum(t.norm(norm_type) ** norm_type
             for t in tensors) ** (1. / norm_type)


class Loss(ABC):
  """ Algorithm loss function. """
  def __init__(self, model, name=None):
    self.model = model
    if name is None:
      name = self.__class__.__name__
      name = name[:-len("Loss")] if name.endswith("Loss") else name
      name = name.lower()
    self.name = name
    self.call_count = 0

  @property
  def device(self):
    """ Returns device of the underlying model. """
    return next(self.model.parameters()).device

  def torch_from_numpy(self, arr):
    """ Casts np.ndarray to torch.Tensor and moves to model device. """
    return torch.from_numpy(arr).to(device=self.device)

  @abstractmethod
  def _compute_loss(self, data):
    """ Implements loss value computation. """

  def __call__(self, data):
    """ Computes and returns loss value on given data. """
    loss = self._compute_loss(data)
    self.call_count += 1
    return loss


class Trainer(ABC):
  """ Class to perform algorithm training. """
  def __init__(self, optimizer, anneals=None, max_grad_norm=None):
    self.optimizer = optimizer
    self.anneals = anneals or []
    self.max_grad_norm = max_grad_norm
    self.step_count = 0

  def preprocess_gradients(self, parameters, name):
    """ Applies gradient preprocessing. """
    grad_norm = None
    if self.max_grad_norm is not None:
      grad_norm = torch.nn.utils.clip_grad_norm_(parameters, self.max_grad_norm)
    if summary.should_record():
      if grad_norm is None:
        grad_norm = total_norm(p.grad for p in parameters if p.grad is not None)
      summary.add_scalar(f"{name}/grad_norm", grad_norm,
                         global_step=self.step_count)

  def step(self, loss, model, name, step_var):
    """ Performs single training step of a given algorithm. """
    loss.backward()
    self.preprocess_gradients(model.parameters(), name)
    for anneal in self.anneals:
      if summary.should_record():
        anneal.summarize(step_var)
      anneal.step_to(int(step_var))
    self.optimizer.step()
    self.optimizer.zero_grad()
    self.step_count += 1
    return loss


class Alg(ABC):
  """ Generic learning algorithm specified by its loss function. """
  def __init__(self, runner, trainer, loss_fn, name=None):
    self.runner = runner
    self.model = self.runner.policy.model
    self.trainer = trainer
    self.loss_fn = loss_fn
    if name is None:
      name = self.__class__.__name__.lower()
    self.name = name

  def loss(self, data):
    """ Computes and returns the loss function on data. """
    return self.loss_fn(data)

  def step(self, data):
    """ Performs learning step of the algorithm. """
    loss = self.loss(data)
    self.trainer.step(loss, self.model, self.name, self.runner.step_count)
    return loss

  def learn(self):
    """ Performs learning with this algorithm. """
    for data in self.runner.run():
      loss = self.step(data)
      yield data, loss
