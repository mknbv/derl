""" Implements Proximal Policy Optimization algorithm.  """
import torch

from derl.alg.common_torch import BaseAlgorithm, r_squared, torch_from_numpy
import derl.summary as summary


class PPO(BaseAlgorithm):
  """ Proximal Policy Optimization algorithm.

  See [Schulman et al., 2017](https://arxiv.org/abs/1707.06347)
  """
  # pylint: disable=too-many-arguments
  def __init__(self,
               policy,
               optimizer=None,
               cliprange=0.2,
               value_loss_coef=0.25,
               entropy_coef=0.01,
               max_grad_norm=0.5,
               step_var=None):
    super().__init__(model=policy.model, optimizer=optimizer, step_var=step_var)
    self.policy = policy
    self.cliprange = cliprange
    self.value_loss_coef = value_loss_coef
    self.entropy_coef = entropy_coef
    self.max_grad_norm = max_grad_norm

  def policy_loss(self, trajectory, act=None):
    """ Compute policy loss (including entropy regularization). """
    if act is None:
      act = self.policy.act(trajectory, training=True)
    if "advantages" not in trajectory:
      raise ValueError("trajectory does not contain 'advantages'")

    old_log_prob = torch_from_numpy(trajectory["log_prob"], self.device)
    advantages = torch_from_numpy(trajectory["advantages"], self.device)
    actions = torch_from_numpy(trajectory["actions"], self.device)

    log_prob = act["distribution"].log_prob(actions)
    if log_prob.shape != old_log_prob.shape:
      raise ValueError("trajectory has mismatched shapes: "
                       f"log_prob.shape={log_prob.shape} "
                       f"old_log_prob.shape={old_log_prob.shape}")
    if log_prob.shape != advantages.shape:
      raise ValueError("trajectory has mismatched shapes: "
                       f"log_prob.shape={log_prob.shape} "
                       f"advantages.shape={advantages.shape}")

    ratio = torch.exp(log_prob - old_log_prob)
    policy_loss = -ratio * advantages
    if self.cliprange is not None:
      ratio_clipped = torch.clamp(ratio, 1. - self.cliprange,
                                  1. + self.cliprange)
      policy_loss_clipped = -ratio_clipped * advantages
      policy_loss = torch.max(policy_loss, policy_loss_clipped)

    policy_loss = torch.mean(policy_loss)
    entropy = torch.mean(act["distribution"].entropy())

    if summary.should_record():
      summaries = dict(advantages=torch.mean(advantages),
                       policy_loss=policy_loss,
                       entropy=entropy)
      for key, val in summaries.items():
        summary.add_scalar(f"ppo/{key}", val, global_step=self.step_var)

    return policy_loss - self.entropy_coef * entropy

  def value_loss(self, trajectory, act=None):
    """ Computes value loss. """
    if act is None:
      act = self.policy.act(trajectory, training=True)
    if "value_targets" not in trajectory:
      raise ValueError("trajectory does not contain 'value_targets'")

    value_targets = torch_from_numpy(trajectory["value_targets"], self.device)
    old_value_preds = torch_from_numpy(trajectory["values"], self.device)
    values = act["values"]

    if values.shape != value_targets.shape:
      raise ValueError("trajectory has mismatched shapes "
                       f"values.shape={values.shape} "
                       f"value_targets.shape={value_targets.shape}")

    value_loss = torch.pow(values - value_targets, 2)
    if self.cliprange is not None:
      values_clipped = old_value_preds + torch.clamp(
          values - old_value_preds, -self.cliprange, self.cliprange)
      value_loss_clipped = torch.pow(values_clipped - value_targets, 2)
      value_loss = torch.max(value_loss, value_loss_clipped)

    value_loss = torch.mean(value_loss)
    if summary.should_record():
      summaries = dict(value_loss=value_loss,
                       value_targets=torch.mean(value_targets),
                       value_preds=torch.mean(values),
                       r_squared=r_squared(value_targets, values))
      for key, val in summaries.items():
        summary.add_scalar(f"ppo/{key}", val, global_step=self.step_var)
    value_loss = torch.mean(value_loss)
    return value_loss

  def loss(self, data):
    """ Returns ppo loss for given data (trajectory dict). """
    act = self.policy.act(data, training=True)
    policy_loss = self.policy_loss(data, act)
    value_loss = self.value_loss(data, act)
    loss = policy_loss + self.value_loss_coef * value_loss
    if summary.should_record():
      summary.add_scalar("ppo/loss", loss, global_step=self.step_var)
    return loss
