""" Implements Soft Actor-Critic algorithm. """
from collections import namedtuple
import torch

from derl.alg.common import Loss, r_squared
from derl.alg.dqn import TargetUpdater
from derl import summary


# pylint: disable=invalid-name
SACLossTuple = namedtuple(
    "SACLossTuple",
    ["policy_loss", "entropy_scale_loss", "qvalue_losses"])
# pylint: enable=invalid-name


class SmoothTargetUpdater(TargetUpdater):
  """ Interface for smoothly updating target model. """
  def __init__(self, model, target, coef=0.005, period=1):
    super().__init__(model, target, period)
    self.coef = coef

  def update(self, step_count):
    for tparam, param in zip(self.target.parameters(),
                             self.model.parameters()):
      tparam.data = self.coef * param + (1 - self.coef) * tparam
    self.last_update_step = step_count


class SACLoss(Loss):
  """ Soft Actor-Critic algorithm loss function.

  See [Haarnoja et al.](https://arxiv.org/abs/1812.05905)
  """
  def __init__(self, policy, target_policy,
               reward_scale=1., gamma=0.99,
               log_entropy_scale=0., name=None):
    super().__init__(model=policy.model, name=name)
    self.policy = policy
    self.target_policy = target_policy
    self.reward_scale = reward_scale
    self.gamma = gamma
    self.log_entropy_scale = log_entropy_scale

  @property
  def entropy_scale(self):
    """ Entropy scale. """
    return torch.exp(self.log_entropy_scale)

  def policy_loss(self, trajectory, act=None):
    """ Computes policy loss. """
    if act is None:
      act = self.policy.act(trajectory, training=True)

    actions = act["sampled_actions"]
    log_prob = act["distribution"].log_prob(actions)
    qvalues = torch.min(torch.stack(act["sampled_actions_qvalues"]), 0).values

    if log_prob.shape + (1,) != qvalues.shape:
      raise ValueError("elements of act have mismatched shapes: "
                       f"log_prob.shape={log_prob.shape}, "
                       f"qvalues.shape={qvalues.shape}; "
                       "expecteed log_prob.shape + (1,) == qvalues.shape")
    qvalues = qvalues.squeeze(-1)

    policy_loss = torch.mean(self.entropy_scale.detach() * log_prob - qvalues)
    if summary.should_record():
      summary.add_scalar(f"{self.name}/policy_loss", policy_loss,
                         global_step=self.call_count)

    return policy_loss

  def entropy_scale_loss(self, trajectory, act=None):
    """ Computes entropy scale loss. """
    if act is None:
      act = self.policy.act(trajectory, training=True)

    actions = act["sampled_actions"]
    log_prob = act["distribution"].log_prob(actions)
    # pylint: disable=not-callable
    target_entropy = -torch.prod(torch.tensor(actions.shape[1:]))
    # pylint: enable=not-callable

    entropy_scale_loss = -self.log_entropy_scale * (log_prob.detach()
                                                    + target_entropy)
    entropy_scale_loss = torch.mean(entropy_scale_loss)

    if summary.should_record():
      summary.add_scalar(f"{self.name}/entropy_scale",
                         self.entropy_scale, global_step=self.call_count)
      summary.add_scalar(f"{self.name}/entropy_scale_loss",
                         entropy_scale_loss, global_step=self.call_count)
    return entropy_scale_loss

  def compute_targets(self, trajectory, act=None):
    """ Computes target values. """
    if act is None:
      act = self.policy.act(trajectory, training=True)

    next_obs = self.torch_from_numpy(trajectory["next_observations"])
    rewards = self.torch_from_numpy(trajectory["rewards"])
    resets = self.torch_from_numpy(trajectory["resets"])

    with torch.no_grad():
      next_actions = act["next_distribution"].sample()
      next_log_prob = act["next_distribution"].log_prob(next_actions)
      next_qvalues = torch.min(torch.stack(
          self.target_policy.act(dict(observations=next_obs,
                                      actions=next_actions),
                                 training=True)["taken_actions_qvalues"]
      ), 0).values
      if next_log_prob.shape + (1,) != next_qvalues.shape:
        raise ValueError("elements of act have mismatched shapes: "
                         f"next_log_prob.shape={next_log_prob.shape}, "
                         f"next_qvalues.shape={next_qvalues.shape}; "
                         "expected next_log_prob.shape + (1,) == "
                         "next_qvalues.shape")
      targets = next_qvalues - self.entropy_scale * next_log_prob[..., None]
      resets = resets.type(targets.dtype)
      targets = (self.reward_scale * rewards
                 + (1 - resets) * self.gamma * targets)
    return targets

  def qvalue_losses(self, trajectory, act=None):
    """ Computes value loss. """
    if act is None:
      act = self.policy.act(trajectory, training=True)

    qtargets = self.compute_targets(trajectory, act)
    qvalues = act["taken_actions_qvalues"]
    qvalue_losses = []
    for i, qpreds in enumerate(qvalues):
      qvalue_losses.append(torch.mean(torch.pow(qtargets - qpreds, 2)))

    if summary.should_record():
      summary.add_scalar(f"{self.name}/qtargets", torch.mean(qtargets),
                         global_step=self.call_count)
      for i, loss in enumerate(qvalue_losses):
        summary.add_scalar(f"{self.name}/qpreds_{i}", torch.mean(qvalues[i]),
                           global_step=self.call_count)
        summary.add_scalar(f"{self.name}/r_squared_{i}",
                           r_squared(qtargets, qvalues[i]),
                           global_step=self.call_count)
        summary.add_scalar(f"{self.name}/qvalue_loss_{i}",
                           loss, global_step=self.call_count)
    return qvalue_losses

  def __call__(self, data):
    act = self.policy.act(data, training=True)
    policy_loss = self.policy_loss(data, act)
    entropy_scale_loss = self.entropy_scale_loss(data, act)
    qvalue_losses = self.qvalue_losses(data, act)
    self.call_count += 1
    return SACLossTuple(policy_loss, entropy_scale_loss, qvalue_losses)
