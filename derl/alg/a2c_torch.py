""" Implements Actor-Critic algorithm. """
import torch
from derl.base_torch import BaseAlgorithm
import derl.summary_manager as summary_manager


def r_squared(targets, predictions):
  """ Computes coefficient of determination. """
  variance = torch.pow(predictions.std(), 2)
  return 1. - torch.mean(torch.pow(predictions - targets, 2)) / variance


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
    self.device = next(self.model.parameters()).device
    self.value_loss_coef = value_loss_coef
    self.entropy_coef = entropy_coef
    self.max_grad_norm = max_grad_norm

  def policy_loss(self, trajectory, act=None):
    """ Compute policiy loss including entropy regularization. """
    if act is None:
      act = self.policy.act(trajectory, training=True)
    log_prob = act["distribution"].log_prob(
        torch.from_numpy(trajectory["actions"]).to(self.device))
    advantages = torch.from_numpy(trajectory["advantages"]).to(self.device)

    if log_prob.shape != advantages.shape:
      raise ValueError("trajectory has mismatched shapes: "
                       f"log_prob.shape={log_prob.shape} "
                       f"advantages.shape={advantages.shape}")

    policy_loss = -torch.mean(log_prob * advantages)
    entropy = torch.mean(act["distribution"].entropy())

    if summary_manager.should_record_summaries():
      summary_manager.add_scalar("a2c/advantages", torch.mean(advantages))
      summary_manager.add_scalar("a2c/entropy", torch.mean(entropy))
      summary_manager.add_scalar("a2c/policy_loss", policy_loss)
    return policy_loss - self.entropy_coef * entropy

  def value_loss(self, trajectory, act=None):
    """ Compute value loss. """
    if act is None:
      act = self.policy.act(trajectory, training=True)
    values = act["values"]
    value_targets = torch.from_numpy(
        trajectory["value_targets"]).to(self.device)

    if values.shape != value_targets.shape:
      raise ValueError("trajectory has mismatched shapes "
                       f"values.shape={values.shape} "
                       f"value_targets.shape={value_targets.shape}")

    value_loss = torch.mean(torch.pow(values - value_targets, 2))

    if summary_manager.should_record_summaries():
      summary_manager.add_scalar("a2c/value_targets",
                                 torch.mean(value_targets))
      summary_manager.add_scalar("a2c/value_preds", torch.mean(values))
      summary_manager.add_scalar("a2c/value_loss", value_loss)
      summary_manager.add_scalar("a2c/r_squared",
                                 r_squared(values, value_targets))

    return value_loss

  def loss(self, data):
    act = self.policy.act(data, training=True)
    policy_loss = self.policy_loss(data, act)
    value_loss = self.value_loss(data, act)
    loss = policy_loss + self.value_loss_coef * value_loss
    if summary_manager.should_record_summaries():
      summary_manager.add_scalar("a2c/loss", loss)
    return loss
