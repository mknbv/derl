""" Deep Q-learning algorithm implementation. """
import tensorflow as tf
from derl.base import BaseAlgorithm
from derl.common import r_squared


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
    self.target_update_period = target_update_period
    self.last_target_update_step = tf.Variable(
        int(self.step_var) - target_update_period,
        dtype=tf.int64, trainable=False, name="last_target_update_step")

  def should_update_target(self):
    """ Returns true if it is time to update target model. """
    return (int(self.step_var) - int(self.last_target_update_step)
            >= self.target_update_period)

  def update_target(self):
    """ Updates target model variables with the trained model variables. """
    self.target_model.set_weights(self.model.get_weights())
    self.last_target_update_step.assign(self.step_var.variable)

  def compute_qvalues(self, observations, actions=None):
    """ Computes qvalues for given observations.

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

  def loss(self, data):
    obs, actions, rewards, resets, next_obs = (data[k] for k in (
        "observations", "actions", "rewards", "resets", "next_observations"))
    next_qvalues = self.compute_qvalues(next_obs)

    if len({rewards.shape, resets.shape}) != 1:
      raise ValueError(
          "rewards, resets must have the same shapes, "
          f"got rewards.shape={rewards.shape}, resets.shape={resets.shape}")
    if tuple(next_qvalues.shape) != resets.shape:
      raise ValueError("argmax-qvalues have bad shape "
                       f"{tuple(next_qvalues.shape)}, expected shape "
                       f"{resets.shape}")

    qtargets = rewards + (1 - resets) * self.gamma * next_qvalues
    qvalues = self.compute_qvalues(obs, actions)
    tf.contrib.summary.scalar("dqn/r_squared", r_squared(qtargets, qvalues),
                              step=self.step_var)
    loss = tf.losses.huber_loss(qtargets, qvalues)
    tf.contrib.summary.scalar("dqn/loss", loss, step=self.step_var)
    return loss

  def preprocess_gradients(self, gradients):
    grad_norm = tf.linalg.global_norm(gradients)
    tf.contrib.summary.scalar("dqn/grad_norm", grad_norm, step=self.step_var)
    return gradients

  def step(self, data):
    if self.should_update_target():
      self.update_target()
    super().step(data)
