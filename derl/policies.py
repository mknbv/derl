""" Reinforcement learning policies. """
from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.distributions import Categorical, Normal, Independent
from torch.distributions.transforms import TanhTransform
from torch.distributions.transformed_distribution import TransformedDistribution


class Policy(ABC):
  """ RL policy (typically wraps a torch.nn.Module).  """
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


def multivariate_normal_diag(loc, scale, axis=1):
  """ Creates multivariate normal distribution with diagonal covariance. """
  return Independent(Normal(loc, scale), axis)


class ActorCriticPolicy(Policy):
  """ Actor-critic policy. """
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
        distribution = multivariate_normal_diag(*distribution_inputs)
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


def tanh_normal_diag(loc, scale, axis=1, cache_size=1):
  """ Creates distribution arising from applying tanh to normal. """
  normal = Independent(Normal(loc, scale), axis)
  transform = TanhTransform(cache_size=cache_size)
  return TransformedDistribution(normal, transform)


class SACPolicy(Policy):
  """ Soft Actor-Critic Policy. """
  def __init__(self, model):
    self.model = model

  def act(self, inputs, state=None, update_state=True, training=False):
    _ = update_state
    if state is not None:
      raise NotImplementedError("SACPolicy does not support state inputs")
    if not training:
      with self.model.policy_context():
        distribution = tanh_normal_diag(*self.model(inputs))
      return dict(actions=_np(distribution.sample()))
    observations, taken_actions = inputs["observations"], inputs["actions"]
    next_observations = inputs.get("next_observations")
    with self.model.qvalues_context():
      taken_actions_qvalues = self.model(observations, taken_actions)
    if next_observations is None:
      return dict(taken_actions_qvalues=taken_actions_qvalues)
    with self.model.policy_context():
      distribution = tanh_normal_diag(*self.model(observations))
      next_distribution = tanh_normal_diag(*self.model(next_observations))
    sampled_actions = distribution.rsample()
    with self.model.qvalues_context():
      sampled_actions_qvalues = self.model(observations, sampled_actions)
    return dict(distribution=distribution,
                taken_actions_qvalues=taken_actions_qvalues,
                sampled_actions=sampled_actions,
                sampled_actions_qvalues=sampled_actions_qvalues,
                next_distribution=next_distribution)


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
      device = next(model.parameters()).device
      vals = torch.linspace(*valrange, model.nbins).to(device)
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
