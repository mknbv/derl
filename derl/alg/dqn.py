""" Deep Q-learning algorithm implementation. """
import tensorflow as tf
from derl.base import BaseAlgorithm
from derl.common import r_squared


class TargetUpdator:
  """ Provides interface for updating target model with a given period. """
  def __init__(self, model, target, step_var, period=40_000):
    self.model = model
    self.target = target
    self.period = period
    self.step_var = step_var
    self.last_update_step = tf.Variable(int(self.step_var) - period,
                                        dtype=tf.int64, trainable=False,
                                        name="last_update_step")

  def should_update(self):
    """ Returns true if it is time to update target model. """
    return int(self.step_var) - int(self.last_update_step) >= self.period

  def update(self):
    """ Updates target model variables with the trained model variables. """
    self.target.set_weights(self.model.get_weights())
    self.last_update_step.assign(self.step_var.variable)


class DQN(BaseAlgorithm):
  """ Deep Q-Learning algorithm.

  See [Mnih et al.](
  https://web.stanford.edu/class/psych209/Readings/MnihEtAlHassibis15NatureControlDeepRL.pdf).
  """
  # pylint: disable=too-many-arguments
  def __init__(self, model, target_model,
               optimizer=None,
               gamma=0.99,
               target_update_period=40_000,
               double=True,
               step_var=None):
    super().__init__(model, optimizer, step_var)
    self.target_model = target_model
    self.gamma = gamma
    self.double = double
    self.target_updator = TargetUpdator(model, target_model,
                                        self.step_var, target_update_period)

  def make_predictions(self, observations, actions=None):
    """ Applies a model to given observations and selects
    predictions based on actions.

    If actions are specified uses training model, otherwise target model.
    If actions are not given uses argmax over training model or
    target model (depending on self.double flag) to compute them.
    """
    if actions is None:
      qvalues = self.target_model(observations)
      actions = tf.cast(tf.argmax(qvalues if not self.double
                                  else self.model(observations), -1), tf.int32)
    else:
      qvalues = self.model(observations)
    indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], -1)
    qvalues = tf.gather_nd(qvalues, indices)
    return qvalues

  def compute_targets(self, rewards, resets, next_obs):
    """ Computes target values. """
    nsteps = rewards.shape[1]
    targets = self.make_predictions(next_obs)

    if len({rewards.shape, resets.shape}) != 1:
      raise ValueError(
          "rewards, resets must have the same shapes, "
          f"got rewards.shape={rewards.shape}, resets.shape={resets.shape}")
    target_shape = rewards.shape[:1] + rewards.shape[2:]
    if tuple(targets.shape) != target_shape:
      raise ValueError("making predictions when computing targets gives bad "
                       f"shape {tuple(targets.shape)}, expected shape "
                       f"{target_shape}")

    for t in reversed(range(nsteps)):
      targets = rewards[:, t] + (1 - resets[:, t]) * self.gamma * targets
    return targets

  def loss(self, data):
    obs, actions, rewards, resets, next_obs = (data[k] for k in (
        "observations", "actions", "rewards", "resets", "next_observations"))

    qtargets = self.compute_targets(rewards, resets, next_obs)
    qvalues = self.make_predictions(obs, actions)
    tf.contrib.summary.scalar("dqn/r_squared", r_squared(qtargets, qvalues),
                              step=self.step_var)
    if "update_priorities" in data:
      data["update_priorities"](tf.abs(qtargets - qvalues).numpy())
    loss = tf.losses.huber_loss(qtargets, qvalues,
                                weights=data.get("weights", 1.))
    tf.contrib.summary.scalar("dqn/loss", loss, step=self.step_var)
    return loss

  def preprocess_gradients(self, gradients):
    grad_norm = tf.linalg.global_norm(gradients)
    tf.contrib.summary.scalar("dqn/grad_norm", grad_norm, step=self.step_var)
    return gradients

  def step(self, data):
    if self.target_updator.should_update():
      self.target_updator.update()
    super().step(data)
