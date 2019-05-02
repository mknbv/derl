""" Imitation learning algorithm. """
import tensorflow as tf

from derl.base import BaseAlgorithm
from derl.common import r_squared


class ActorCriticImitation(BaseAlgorithm):
  """ Imitation learning algorithm. """
  def __init__(self,
               train_policy,
               target_policy,
               optimizer=None,
               value_loss_coef=0.25,
               entropy_coef=0.,
               step_var=None):
    super().__init__(train_policy.model, optimizer=optimizer,
                     step_var=step_var)
    self.train_policy = train_policy
    self.target_policy = target_policy
    self.value_loss_coef = value_loss_coef
    self.entropy_coef = entropy_coef

  def _get_act(self, data, act=None):
    """ Creates act if it is None, otherwise just returns it. """
    if act is None:
      act_train = self.train_policy.act(data, training=True)
      act_target = self.target_policy.act(data, training=True)
      return act_train, act_target
    return act

  def policy_loss(self, data, act=None):
    """ Computes and returns policy loss. """
    act_train, act_target = self._get_act(data, act)
    dtrain, dtarget = act_train["distribution"], act_target["distribution"]

    cross_entropy = tf.reduce_mean(dtarget.cross_entropy(dtrain))
    tf.contrib.summary.scalar("imitation/cross_entropy", cross_entropy,
                              step=self.step_var)
    entropy = tf.reduce_mean(dtrain.entropy())
    tf.contrib.summary.scalar("imitation/entropy", entropy, step=self.step_var)

    policy_loss = cross_entropy + self.entropy_coef * entropy
    tf.contrib.summary.scalar("imitation/policy_loss", policy_loss,
                              step=self.step_var)
    return policy_loss

  def value_loss(self, data, act=None):
    """ Computes and returns value loss. """
    act_train, act_target = self._get_act(data, act)
    train_values, target_values = act_train["values"], act_target["values"]
    if train_values.shape != target_values.shape:
      raise ValueError("mismatched shapes: train policy has values "
                       f"with shape {train_values.shape}, target policy "
                       f"has values with shape {target_values.shape}")
    tf.contrib.summary.scalar(
        "imitation/r_squared", r_squared(train_values, target_values),
        step=self.step_var)
    value_loss = tf.reduce_mean(tf.square(train_values - target_values))
    tf.contrib.summary.scalar("imitation/value_loss", value_loss,
                              step=self.step_var)
    return value_loss

  def loss(self, data):
    act = self._get_act(data)
    policy_loss = self.policy_loss(data, act)
    value_loss = self.value_loss(data, act)
    return policy_loss + self.value_loss_coef * value_loss
