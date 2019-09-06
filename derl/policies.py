""" Reinforcement learning policies. """
from abc import ABC, abstractmethod

import numpy as np
import tensorflow as tf
import tensorflow_probability as tfp


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


def _call_model(model, inputs):
  """ Calls model possibly with input broadcasting. """
  expand_dims = model.input.shape.ndims - inputs.ndim
  inputs = inputs[(None,) * expand_dims]
  outputs = model(inputs)
  squeeze_dims = tuple(range(expand_dims))
  if squeeze_dims:
    if isinstance(outputs, (list, tuple)):
      return type(outputs)(map(lambda t: tf.squeeze(t, squeeze_dims), outputs))
    return tf.squeeze(outputs, squeeze_dims)
  return outputs


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

    *distribution_inputs, values = _call_model(self.model, observations)
    if self.distribution is None:
      if len(distribution_inputs) == 1:
        distribution = tfp.distributions.Categorical(*distribution_inputs)
      elif len(distribution_inputs) == 2:
        distribution = tfp.distributions.MultivariateNormalDiag(
            *distribution_inputs)
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
    return {"actions": actions.numpy(),
            "log_prob": log_prob.numpy(),
            "values": values.numpy()}


class EpsilonGreedyPolicy(Policy):
  """ Epsilon greedy policy. """
  def __init__(self, model, epsilon=0.05):
    self.model = model
    self.epsilon = epsilon

  def act(self, inputs, state=None, update_state=True, training=False):
    if state is not None:
      raise ValueError("epsilon greedy policy does not support state inputs")

    epsilon = self.epsilon
    if isinstance(epsilon, (tf.Tensor, tf.Variable)):
      epsilon = epsilon.numpy()
    if np.random.random() <= epsilon:
      return {"actions": np.random.randint(self.model.output.shape[1].value)}

    preds = _call_model(self.model, inputs).numpy()
    action_dims = preds.ndim - (inputs.ndim == self.model.input.shape.ndims)
    qvalues = np.mean(preds, -1 if action_dims > 1 else ())
    return {"actions": np.argmax(qvalues, -1)}
