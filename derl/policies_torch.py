""" Reinforcement learning policies. """
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal


class Policy(ABC):
  """ RL policy (typically wraps a keras model).  """
  def is_recurrent(self): # pylint: disable=no-self-use
    """ Returns true if policy is recurrent. """
    return False

  def get_state(self): # pylint: disable=no-self-use
    """ Returns current policy state. """
    return None

  def reset(self): # pylint: disable=no-self-use
    """ Resets the state. """

  @abstractmethod
  def act(self, inputs, state=None, update_state=True, training=False):
    """ Returns `dict` of all the outputs of the policy.

    If `training=False`, then inputs can be a batch of observations
    or a `dict` containing `observations` key. Otherwise,
    `inputs` should be a trajectory dictionary with all keys
    necessary to recompute outputs for training.
    """


def _np(tensor):
  """ Converts given tensor to numpy array. """
  return tensor.cpu().detach().numpy()


class ActorCriticPolicy(Policy):
  """ Actor critic policy with discrete number of actions. """
  def __init__(self, model, distribution=None):
    self.model = model
    self.distribution = distribution

  def act(self, inputs, state=None, update_state=True, training=False):
    # TODO: support recurrent policies.
    _ = update_state
    if state is not None:
      raise NotImplementedError()
    if training:
      observations = inputs["observations"]
    else:
      observations = inputs

    *distribution_inputs, values = self.model(observations)
    if self.distribution is None:
      if len(distribution_inputs) == 1:
        distribution = Categorical(logits=distribution_inputs[0])
      elif len(distribution_inputs) == 2:
        loc = distribution_inputs[0]
        scale_tril = torch.diag(distribution_inputs[1])
        distribution = MultivariateNormal(loc=loc, scale_tril=scale_tril)
      else:
        raise ValueError(f"model has {len(distribution_inputs)} "
                         "outputs to create a distribution, "
                         "expected a single output for categorical "
                         "and two outputs for normal distributions")
    else:
      distribution = self.distribution(*distribution_inputs)
    if training:
      return {"distribution": distribution, "values": values}
    actions = distribution.sample()
    log_prob = distribution.log_prob(actions)
    return {"actions": _np(actions),
            "log_prob": _np(log_prob),
            "values": _np(values)}


class EpsilonGreedyPolicy(Policy):
  """ Epsilon greedy policy. """
  def __init__(self, model, epsilon=0.05, nactions=None,
               qvalues_from_preds=None):
    self.model = model
    self.epsilon = epsilon
    self.nactions = nactions
    if qvalues_from_preds is None:
      qvalues_from_preds = lambda x: x
    self.qvalues_from_preds = qvalues_from_preds

  @classmethod
  def categorical(cls, model, epsilon=0.05, nactions=None,
                  valrange=(-10., 10.)):
    """ Categorical distributional RL policy. """
    def qvalues_from_preds(preds):
      vals = torch.linspace(*valrange, model.nbins)
      return torch.sum(vals * preds, -1)
    return cls(model, epsilon, nactions, qvalues_from_preds)

  @classmethod
  def quantile(cls, model, epsilon=0.05, nactions=None):
    """ Quantile distributional RL policy. """
    def qvalues_from_preds(preds):
      return torch.mean(preds, -1)
    return cls(model, epsilon, nactions, qvalues_from_preds)

  def act(self, inputs, state=None, update_state=True, training=False):
    if state is not None:
      raise ValueError("epsilon greedy policy does not support state inputs")

    epsilon = self.epsilon
    if isinstance(epsilon, (torch.Tensor)):
      epsilon = epsilon.numpy()
    if self.nactions is None:
      preds = self.model(inputs)
      qvalues = self.qvalues_from_preds(preds)
      self.nactions = qvalues.shape[-1]
    if not training and np.random.random() <= epsilon:
      return {"actions": np.random.randint(self.nactions)}

    preds = self.model(inputs)
    if training:
      return dict(preds=preds)

    qvalues = self.qvalues_from_preds(preds)
    actions = _np(torch.argmax(qvalues, -1))
    return dict(actions=actions)
