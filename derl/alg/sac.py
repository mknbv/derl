""" Implements Soft Actor-Critic algorithm. """
from collections import namedtuple
from copy import deepcopy
from itertools import chain
import torch

from derl.alg.common import Trainer, Alg, Loss, r_squared
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
      if qtargets.shape != qpreds.shape:
        raise ValueError("qpreds and qtargets have mismatched shapes, "
                         f"act['taken_actions_qvalues'][{i}]={qpreds.shape} "
                         f"qtargets.shape={qtargets.shape}")
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


def reset_model(model):
  """ Reinitializes the layers of the model. """
  if hasattr(model, "init_fn"):
    model.apply(model.init_fn)
    return
  for child in model.children():
    reset_model(child)


class SAC(Alg):
  """ Soft Actor-Critic algorithm.

  See [Haarnoja et al.](https://arxiv.org/abs/1812.05905)
  """
  def __init__(self, runner, trainer,
               target_policy=None, target_updater=None,
               name=None, **loss_kwargs):
    if target_policy is None:
      target_policy = deepcopy(runner.policy)
      reset_model(target_policy.model)
    loss_fn = SACLoss(runner.policy, target_policy, name=name, **loss_kwargs)
    super().__init__(runner, trainer, loss_fn, name=name)
    self.target_updater = target_updater
    self.step_count = 0

  @classmethod
  def make(cls, runner, trainer, target_policy=None,
           target_update_coef=0.005, target_update_period=1,
           **kwargs):
    """ Creates SAC algorithm with target updater from arguments. """
    model = runner.policy.model
    if target_policy is None:
      target_policy = deepcopy(runner.policy)
    target_updater = SmoothTargetUpdater(
        model, target_policy.model, target_update_coef, target_update_period)
    return cls(runner, trainer, target_policy, target_updater, **kwargs)

  def step(self, data):
    if self.target_updater.should_update(self.step_count):
      self.target_updater.update(self.step_count)
    loss = super().step(data)
    self.step_count += 1
    return loss


class SACTrainer(Trainer):
  """ SAC trainer. """
  def __init__(self, policy_opt, entropy_scale_opt, qvalue_opts,
               max_grad_norm=None):
    super().__init__(optimizer=None, max_grad_norm=max_grad_norm)
    self.policy_opt = policy_opt
    self.entropy_scale_opt = entropy_scale_opt
    self.qvalue_opts = qvalue_opts

  def optimizer_step(self, optimizer, loss, tag):
    """ Performs single step of the optimizer. """
    optimizer.zero_grad()
    loss.backward()
    self.preprocess_gradients(chain.from_iterable(
        group["params"] for group in optimizer.param_groups), tag)
    optimizer.step()

  def step_policy(self, alg, *, policy_loss):
    """ Performs policy update. """
    self.optimizer_step(self.policy_opt, policy_loss,
                        f"{alg.name}/policy_grad_norm")

  def step_entropy_scale(self, alg, *, entropy_scale_loss):
    """ Performs entropy scale update. """
    self.optimizer_step(self.entropy_scale_opt, entropy_scale_loss,
                        f"{alg.name}/entropy_scale_grad_norm")

  def step_qvalues(self, alg, *, qvalue_losses):
    """ Performs qvalue updates. """
    if len(self.qvalue_opts) != len(qvalue_losses):
      raise ValueError("number of qvalue optimizers "
                       f"({len(self.qvalue_opts)}) does not match the "
                       f"number of losses ({len(qvalue_losses)})")
    for i, (opt, loss) in enumerate(zip(self.qvalue_opts, qvalue_losses)):
      self.optimizer_step(opt, loss, f"{alg.name}/qvalues_{i}_grad_norm")

  def step(self, alg, data):
    loss = alg.loss(data)
    self.step_policy(alg, policy_loss=loss.policy_loss)
    self.step_entropy_scale(alg, entropy_scale_loss=loss.entropy_scale_loss)
    self.step_qvalues(alg, qvalue_losses=loss.qvalue_losses)
    self.step_count += 1
    return loss
