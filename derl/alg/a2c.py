""" Implements Actor-Critic algorithm. """
import torch
from derl.alg.common import Loss, Alg, r_squared
import derl.summary as summary


class A2CLoss(Loss):
  """ Advantage Actor Critic loss.

  See [Sutton et al., 1998](http://incompleteideas.net/book/the-book-2nd.html).
  """
  def __init__(self, policy, value_loss_coef=0.25, entropy_coef=0.01,
               name=None):
    super().__init__(model=policy.model, name=name)
    self.policy = policy
    self.value_loss_coef = value_loss_coef
    self.entropy_coef = entropy_coef

  def policy_loss(self, trajectory, act=None):
    """ Compute policiy loss including entropy regularization. """
    if act is None:
      act = self.policy.act(trajectory, training=True)
    log_prob = act["distribution"].log_prob(
        self.torch_from_numpy(trajectory["actions"]))
    advantages = self.torch_from_numpy(trajectory["advantages"])

    if log_prob.shape != advantages.shape:
      raise ValueError("trajectory has mismatched shapes: "
                       f"log_prob.shape={log_prob.shape} "
                       f"advantages.shape={advantages.shape}")

    policy_loss = -torch.mean(log_prob * advantages)
    entropy = torch.mean(act["distribution"].entropy())

    if summary.should_record():
      summaries = dict(advantages=torch.mean(advantages),
                       entropy=torch.mean(entropy),
                       policy_loss=policy_loss)
      for key, val in summaries.items():
        summary.add_scalar(f"{self.name}/{key}", val,
                           global_step=self.call_count)

    return policy_loss - self.entropy_coef * entropy

  def value_loss(self, trajectory, act=None):
    """ Compute value loss. """
    if act is None:
      act = self.policy.act(trajectory, training=True)
    values = act["values"]
    value_targets = self.torch_from_numpy(trajectory["value_targets"])

    if values.shape != value_targets.shape:
      raise ValueError("trajectory has mismatched shapes "
                       f"values.shape={values.shape} "
                       f"value_targets.shape={value_targets.shape}")

    value_loss = torch.mean(torch.pow(values - value_targets, 2))

    if summary.should_record():
      summaries = dict(value_targets=torch.mean(value_targets),
                       value_preds=torch.mean(values),
                       value_loss=value_loss,
                       r_squared=r_squared(values, value_targets))
      for key, val in summaries.items():
        summary.add_scalar(f"{self.name}/{key}", val,
                           global_step=self.call_count)

    return value_loss

  def _compute_loss(self, data):
    act = self.policy.act(data, training=True)
    policy_loss = self.policy_loss(data, act)
    value_loss = self.value_loss(data, act)
    loss = policy_loss + self.value_loss_coef * value_loss
    if summary.should_record():
      summary.add_scalar(f"{self.name}/loss", loss,
                         global_step=self.call_count)
    return loss


class A2C(Alg):
  """ Advantage Actor Critic.

  See [Sutton et al., 1998](http://incompleteideas.net/book/the-book-2nd.html).
  """
  def __init__(self, runner, trainer, value_loss_coef=0.25,
               entropy_coef=0.01, name=None):
    loss_fn = A2CLoss(runner.policy, value_loss_coef=value_loss_coef,
                      entropy_coef=entropy_coef, name=name)
    super().__init__(runner, trainer, loss_fn, name=name)
