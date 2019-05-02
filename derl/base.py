"""
Defines base classes.
"""
from abc import ABC, abstractmethod
import os
import re

import tensorflow as tf
from tqdm import tqdm
from .train import StepVariable


class BaseRunner(ABC):
  """ General data runner. """
  def __init__(self, env, policy, step_var=None):
    self.env = env
    self.policy = policy
    if step_var is None:
      step_var = StepVariable(f"{camel2snake(self.__class__.__name__)}_step",
                              tf.train.create_global_step())
    self.step_var = step_var

  @property
  def nenvs(self):
    """ Returns number of batched envs or `None` if env is not batched """
    return getattr(self.env.unwrapped, "nenvs", None)

  @abstractmethod
  def get_next(self):
    """ Returns next data object """


def camel2snake(name):
  """ Converts camel case to snake case. """
  sub = re.sub('(.)([A-Z][a-z]+)', r'\1_\2', name)
  return re.sub('([a-z0-9])([A-Z])', r'\1_\2', sub).lower()


class BaseAlgorithm(ABC):
  """ Base algorithm. """
  def __init__(self, model, optimizer=None, step_var=None):
    self.model = model
    self.optimizer = optimizer or self.model.optimizer
    if step_var is None:
      step_var = StepVariable(f"{camel2snake(self.__class__.__name__)}_step")
    self.step_var = step_var

  @abstractmethod
  def loss(self, data):
    """ Computes the loss given inputs and target values. """

  def preprocess_gradients(self, gradients):
    """ Applies gradient preprocessing. """
    # pylint: disable=no-self-use
    return gradients

  def step(self, data):
    """ Performs single training step of the algorithm. """
    with tf.GradientTape() as tape:
      loss = self.loss(data)
    gradients = self.preprocess_gradients(
        tape.gradient(loss, self.model.trainable_variables))
    self.optimizer.apply_gradients(zip(gradients,
                                       self.model.trainable_variables))
    if getattr(self.step_var, "auto_update", True):
      self.step_var.assign_add(1)
    return loss


class Learner:
  """ High-level class for performing learning. """
  def __init__(self, runner, alg):
    self.runner = runner
    self.alg = alg

  @staticmethod
  def make_runner(env, args, model=None):
    """ Creates a runner based on the argparse Namespace. """
    raise NotImplementedError("Learner does not implement make_runner method")

  @staticmethod
  def make_alg(runner, args):
    """ Creates learner algorithm. """
    raise NotImplementedError("Learner does not implement make_alg method")

  @property
  def model(self):
    """ Model trained by the algorithm. """
    return self.alg.model

  @classmethod
  def from_env_args(cls, env, args, model=None):
    """ Creates a learner instance from environment and args namespace. """
    runner = cls.make_runner(env, args, model=model)
    return cls(runner, cls.make_alg(runner, args))

  def learn(self, nsteps, logdir=None, log_period=1, save_weights=None):
    """ Performs learning for a specified number of steps. """
    if not getattr(self.runner.step_var, "auto_update", True):
      raise ValueError("learn method is not supported when runner.step_var "
                       "does not auto-update")
    if save_weights and logdir is None:
      raise ValueError("logdir cannot be None when save_weights is True")
    if save_weights is None:
      save_weights = logdir is not None

    if logdir is not None:
      summary_writer = tf.contrib.summary.create_file_writer(logdir)
      summary_writer.set_as_default()
    step = self.runner.step_var
    if isinstance(step, StepVariable):
      step = step.variable
    with tqdm(total=nsteps) as pbar,\
        tf.contrib.summary.record_summaries_every_n_global_steps(
            log_period, global_step=step):
      while int(self.runner.step_var) < nsteps:
        pbar.update(int(self.runner.step_var) - pbar.n)
        data = self.runner.get_next()
        self.alg.step(data)
    if save_weights:
      self.model.save_weights(os.path.join(logdir, "model"))
