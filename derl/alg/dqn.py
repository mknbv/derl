""" Deep Q-learning algorithm implementation. """
import torch
import torch.nn.functional as F
from derl.alg.common import Loss, Alg, r_squared
import derl.summary as summary


class TargetUpdater:
  """ Provides interface for updating target model with a given period. """
  def __init__(self, model, target, period=10_000):
    self.model = model
    self.target = target
    if period is None:
      period = float("inf")
    self.period = period
    self.last_update_step = -float("inf")

  def should_update(self, step_count):
    """ Returns true if it is time to update target model. """
    return step_count - self.last_update_step >= self.period

  def update(self, step_count):
    """ Updates target model variables with the trained model variables. """
    self.target.load_state_dict(self.model.state_dict())
    self.last_update_step = step_count


def huber_loss(predictions, targets, weights=None):
  """ Huber loss with weights for each element in batch. """
  if weights is None:
    return F.smooth_l1_loss(predictions, targets)
  losses = F.smooth_l1_loss(predictions, targets, reduction=None)
  return torch.mean(weights * losses)


class DQNLoss(Loss):
  """ Deep Q-Learning algorithm loss function.

  See [Mnih et al.](
  https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf).
  """
  def __init__(self, model, target_model, gamma=0.99, double=True, name=None):
    super().__init__(model, name)
    self.target_model = target_model
    self.gamma = gamma
    self.double = double

  def make_predictions(self, observations, actions=None):
    """ Applies a model to given observations and selects
    predictions based on actions.

    If actions are specified uses training model, otherwise target model.
    If actions are not given uses argmax over training model or
    target model (depending on self.double flag) to compute them.
    """
    if actions is None:
      with torch.no_grad():
        qvalues = self.target_model(observations)
        actions = torch.argmax(qvalues if not self.double
                               else self.model(observations), -1)
    else:
      qvalues = self.model(observations)
    qvalues = torch.gather(qvalues, 1, actions[:, None]).squeeze(1)
    return qvalues

  def compute_targets(self, rewards, resets, next_obs):
    """ Computes target values. """
    nsteps = rewards.shape[1]
    targets = self.make_predictions(next_obs)
    resets = resets.type(targets.dtype)

    if len({rewards.shape, resets.shape}) != 1:
      raise ValueError(
          "rewards, resets must have the same shapes, "
          f"got rewards.shape={rewards.shape}, resets.shape={resets.shape}")
    target_shape = rewards.shape[:1] + rewards.shape[2:]
    if tuple(targets.shape) != target_shape:
      raise ValueError("making predictions when computing targets gives bad "
                       f"shape {tuple(targets.shape)}, expected shape "
                       f"{tuple(target_shape)}")

    for t in reversed(range(nsteps)):
      targets = rewards[:, t] + (1 - resets[:, t]) * self.gamma * targets
    return targets

  def _compute_loss(self, data):
    obs, actions, rewards, resets, next_obs = (
        self.torch_from_numpy(data[k]) for k in (
            "observations", "actions", "rewards",
            "resets", "next_observations"))

    qtargets = self.compute_targets(rewards, resets, next_obs)
    qvalues = self.make_predictions(obs, actions)
    if "update_priorities" in data:
      data["update_priorities"](
          torch.abs(qtargets - qvalues).cpu().detach().numpy())

    weights = None
    if "weights" in data:
      weights = self.torch_from_numpy(data["weights"])
    loss = huber_loss(qtargets, qvalues, weights=weights)

    if summary.should_record():
      summary.add_scalar("dqn/r_squared", r_squared(qtargets, qvalues),
                         global_step=self.call_count)
      summary.add_scalar("dqn/loss", loss, global_step=self.call_count)
    return loss


class DQN(Alg):
  """ Deep Q-Learning algorithm.

  See [Mnih et al.](
  https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf).
  """
  def __init__(self, runner, trainer,
               target_model=None, target_update_period=10_000,
               name=None, **loss_kwargs):
    model = runner.policy.model
    if target_model is None:
      target_model = runner.policy.model
      target_update_period = None
    loss_fn = DQNLoss(runner.policy.model, target_model,
                      name=name, **loss_kwargs)
    super().__init__(runner, trainer, loss_fn, name=name)
    self.target_updater = TargetUpdater(model, target_model,
                                        target_update_period)
    self.step_count = 0

  def step(self, data):
    if self.target_updater.should_update(self.step_count):
      self.target_updater.update(self.step_count)
    super().step(data)
    self.step_count += 1
