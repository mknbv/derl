""" Implements Actor-Critic algorithm. """
import tensorflow as tf
from derl.base import BaseAlgorithm
from derl.common import (
    r_squared, reduce_add_summary, maybe_clip_by_global_norm_with_summary)


class A2C(BaseAlgorithm):
  """ Advantage Actor Critic.

  See [Sutton et al., 1998](http://incompleteideas.net/book/the-book-2nd.html).
  """
  def __init__(self,
               policy,
               optimizer,
               value_loss_coef=0.25,
               entropy_coef=0.01,
               max_grad_norm=0.5,
               step_var=None):
    super().__init__(model=policy.model, optimizer=optimizer, step_var=step_var)
    self.policy = policy
    self.value_loss_coef = value_loss_coef
    self.entropy_coef = entropy_coef
    self.max_grad_norm = max_grad_norm

  def policy_loss(self, trajectory, act=None):
    """ Computes policy loss (including entropy regularization). """
    if act is None:
      act = self.policy.act(trajectory, training=True)
    log_prob = act["distribution"].log_prob(trajectory["actions"])
    advantages = trajectory["advantages"]

    if log_prob.shape != advantages.shape:
      raise ValueError("trajectory has mismatched shapes: "
                       f"log_prob.shape={log_prob.shape} "
                       f"advantages.shape={advantages.shape}")

    policy_loss = -tf.reduce_mean(log_prob * advantages)
    entropy = tf.reduce_mean(act["distribution"].entropy())
    reduce_add_summary("a2c/advantages", advantages, step=self.step_var)
    tf.contrib.summary.scalar("a2c/policy_loss", policy_loss,
                              step=self.step_var)
    tf.contrib.summary.scalar("a2c/entropy", entropy, step=self.step_var)
    return policy_loss - self.entropy_coef * entropy

  def value_loss(self, trajectory, act=None):
    """ Computes value loss. """
    if act is None:
      act = self.policy.act(trajectory, training=True)
    values = act["values"]
    value_targets = trajectory["value_targets"]

    if values.shape != value_targets.shape:
      raise ValueError("trajectory has mismatched shapes "
                       f"values.shape={values.shape} "
                       f"value_targets.shape={value_targets.shape}")

    value_loss = tf.reduce_mean(tf.square(values - value_targets))
    reduce_add_summary("a2c/value_targets", value_targets, step=self.step_var)
    reduce_add_summary("a2c/value_preds", values, step=self.step_var)
    tf.contrib.summary.scalar("a2c/value_loss", value_loss, step=self.step_var)
    tf.contrib.summary.scalar("a2c/r_squared", r_squared(value_targets, values),
                              step=self.step_var)
    return value_loss

  def loss(self, data):
    act = self.policy.act(data, training=True)
    policy_loss = self.policy_loss(data, act)
    value_loss = self.value_loss(data, act)
    loss = policy_loss + self.value_loss_coef * value_loss
    tf.contrib.summary.scalar("a2c/loss", loss, step=self.step_var)
    return loss

  def preprocess_gradients(self, gradients):
    return maybe_clip_by_global_norm_with_summary(
        "a2c/grad_norm", gradients, self.max_grad_norm, step=self.step_var)
