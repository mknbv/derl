""" Impelments distributional Q-learning with quantile regression. """
import numpy as np
import tensorflow as tf
from derl.alg.dqn import DQN
from derl.common import r_squared


def qr_dqn_loss(targets, preds, huber_loss=True):
  """ Computes QR-DQN loss given targets and predictions. """
  if targets.ndim != 2 or targets.shape != preds.shape:
    raise ValueError(f"invalid shape(s) targets.shape={targets.shape}"
                     f"preds.shape={preds.shape}, expected 2-d "
                     "tensors of the same shape")
  batch_size, dist_size = targets.shape.as_list()

  cdf = np.arange(0, dist_size + 1) / dist_size
  midpoints = (cdf[:-1] + cdf[1:]) / 2
  overestimation = tf.cast(targets[..., None] < preds[:, None], tf.float32)

  targets = tf.broadcast_to(targets[..., None], (batch_size, dist_size,
                                                 dist_size))
  preds = tf.broadcast_to(preds[:, None], (batch_size, dist_size, dist_size))
  delta_fn = (tf.losses.huber_loss if huber_loss else
              tf.losses.absolute_difference)
  delta = delta_fn(targets, preds, reduction=tf.losses.Reduction.NONE)
  weights = tf.abs(midpoints[None, None] - overestimation)
  loss = tf.reduce_sum(tf.reduce_mean(delta * weights, axis=[0, 1]))
  return loss


class QR_DQN(DQN): # pylint: disable=invalid-name
  """ Distributional Q-learning with quantile regression algorithm.

  See [Dabney et al.](https://arxiv.org/abs/1710.10044).
  """
  # pylint: disable=too-many-arguments
  def __init__(self, model, target_model, optimizer=None,
               gamma=0.99,
               target_update_period=40_000,
               double=True,
               huber_loss=True,
               step_var=None):
    super().__init__(model, target_model, optimizer, gamma,
                     target_update_period, double, step_var)
    self.huber_loss = huber_loss

  def make_predictions(self, observations, actions=None):
    if actions is None:
      qdistribution = self.target_model(observations)
      qvalues = tf.reduce_mean(self.model(observations) if self.double
                               else qdistribution, -1)
      actions = tf.cast(tf.argmax(qvalues, -1), tf.int32)
    else:
      qdistribution = self.model(observations)
    indices = tf.stack([tf.range(tf.shape(actions)[0]), actions], -1)
    return tf.gather_nd(qdistribution, indices)

  def loss(self, data):
    obs, actions, rewards, resets, next_obs = (data[k] for k in (
        "observations", "actions", "rewards", "resets", "next_observations"))

    nbins = self.model.output.shape[-1].value
    rewards = np.tile(rewards[..., None], nbins)
    resets = np.tile(resets[..., None], nbins)
    target_dist = self.compute_targets(rewards, resets, next_obs)
    dist = self.make_predictions(obs, actions)
    tf.contrib.summary.scalar(
        "qr_dqn/r_squared", r_squared(tf.reduce_mean(target_dist, -1),
                                      tf.reduce_mean(dist, -1)),
        step=self.step_var)

    loss = qr_dqn_loss(target_dist, dist, huber_loss=self.huber_loss)
    tf.contrib.summary.scalar("qr_dqn/loss", loss, step=self.step_var)
    return loss

  def preprocess_gradients(self, gradients):
    grad_norm = tf.linalg.global_norm(gradients)
    tf.contrib.summary.scalar("qr_dqn/grad_norm", grad_norm, step=self.step_var)
    return gradients
