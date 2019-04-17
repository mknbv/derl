""" Keras models for reinforcement learning problems. """
from math import sqrt
import gym.spaces
import tensorflow as tf
from .common import clone_model
from .env.env_batch import SpaceBatch


def maybe_rescale(inputs, ubyte_rescale=None):
  """ Rescales inputs to [0, 1] if they are in uint8 and flag not False. """
  if ubyte_rescale and inputs.dtype != tf.uint8.as_numpy_dtype:
    raise ValueError("ubyte_rescale was set to True but "
                     f"inputs.dtype is {inputs.dtype}")
  if (ubyte_rescale or
      ubyte_rescale is None and inputs.dtype == tf.uint8.as_numpy_dtype):
    return tf.cast(inputs, tf.float32) / 255.
  return inputs


class MaybeRescale(tf.keras.layers.Lambda):
  """ Rescales inputs to [0, 1] if they are in uint8 and flag not False. """
  def __init__(self, input_shape, ubyte_rescale=None):
    super().__init__(
        lambda inputs: maybe_rescale(inputs, ubyte_rescale),
        input_shape=input_shape,
        # Specifying output_shape is necessary to prevent redundant call
        # that will determine the output shape by calling the function
        # with tf.float32 type tensor
        output_shape=input_shape
    )


class NatureDQNBase(tf.keras.models.Sequential):
  """ Hidden layers of the Nature DQN model. """
  def __init__(self,
               input_shape=(84, 84, 4),
               ubyte_rescale=None,
               kernel_initializer=tf.initializers.orthogonal(sqrt(2)),
               bias_initializer=tf.initializers.zeros()):
    super().__init__([
        MaybeRescale(input_shape=input_shape,
                     ubyte_rescale=ubyte_rescale),
        tf.keras.layers.Conv2D(
            filters=32,
            kernel_size=8,
            strides=4,
            activation=tf.nn.relu,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer,
        ),
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=4,
            strides=2,
            activation=tf.nn.relu,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer
        ),
        tf.keras.layers.Conv2D(
            filters=64,
            kernel_size=3,
            activation=tf.nn.relu,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(
            units=512,
            activation=tf.nn.relu,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer
        ),
    ])


class NatureDQNModel(tf.keras.Model):
  """ Nature DQN model with possibly several outputs. """
  # pylint: disable=too-many-arguments
  def __init__(self,
               output_units,
               input_shape=(84, 84, 4),
               ubyte_rescale=None,
               kernel_initializer=tf.initializers.orthogonal(sqrt(2)),
               bias_initializer=tf.initializers.zeros()):
    base = NatureDQNBase(input_shape, ubyte_rescale,
                         kernel_initializer, bias_initializer)
    inputs = tf.keras.layers.Input(input_shape)
    base_outputs = base(inputs)
    if isinstance(output_units, (list, tuple)):
      outputs = [
          tf.keras.layers.Dense(
              units=units,
              kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer)(base_outputs)
          for units in output_units
      ]
    else:
      outputs = tf.keras.layers.Dense(
          units=output_units,
          kernel_initializer=kernel_initializer,
          bias_initializer=bias_initializer)(base_outputs)
    super().__init__(inputs=inputs, outputs=outputs)


class MLPBase(tf.keras.Sequential):
  """ MLP model that could be used in classic control or mujoco envs. """
  def __init__(self,
               nlayers=2,
               hidden_units=64,
               activation=tf.nn.tanh,
               kernel_initializer=tf.initializers.orthogonal(sqrt(2))):
    super().__init__([
        tf.keras.layers.Dense(
            units=hidden_units,
            activation=activation,
            kernel_initializer=kernel_initializer
        ) for _ in range(nlayers)
    ])


class MLPModel(tf.keras.Model):
  """ MLP model for given action space. """
  # pylint: disable=too-many-arguments
  def __init__(self, input_shape, output_units, base=None, copy=True,
               kernel_initializer=tf.initializers.orthogonal(sqrt(2)),
               bias_initializer=tf.initializers.zeros()):
    if base is None:
      base = MLPBase()
    if not isinstance(output_units, (list, tuple)):
      output_units = [output_units]

    inputs = tf.keras.layers.Input(input_shape)
    base_outputs = base(inputs)
    outputs = []
    for nunits in output_units:
      outputs.append(
          tf.keras.layers.Dense(
              units=nunits,
              kernel_initializer=kernel_initializer,
              bias_initializer=bias_initializer)(base_outputs)
      )
      if copy:
        base_outputs = clone_model(base)(inputs)
    super().__init__(inputs=inputs, outputs=outputs)


class MujocoModel(tf.keras.Model):
  """ Typical model trained in MuJoCo environments. """
  def __init__(self, input_shape, output_units):
    super().__init__()
    if isinstance(output_units, int):
      output_units = [output_units]
    nactions = output_units[0]
    self.logstd = tf.Variable(tf.zeros(nactions), trainable=True, name="logstd")
    self.model = MLPModel(input_shape, output_units)

  @property
  def input(self):
    return self.model.input

  def call(self, inputs): # pylint: disable=arguments-differ
    inputs = tf.cast(inputs, tf.float32)
    logits, *outputs = self.model(inputs)
    std = tf.exp(self.logstd)
    std = tf.tile(std[None], [inputs.shape[0].value, 1])
    return (logits, std, *outputs)


def make_model(observation_space, action_space, other_outputs=None):
  """ Creates one of default models for a given action space. """
  if isinstance(other_outputs, int) or other_outputs is None:
    other_outputs = [other_outputs] if other_outputs is not None else []

  if isinstance(action_space, SpaceBatch):
    action_space = action_space.spaces[0]
  if isinstance(action_space, gym.spaces.Discrete):
    output_units = [action_space.n, *other_outputs]
    return NatureDQNModel(input_shape=observation_space.shape,
                          output_units=output_units)
  if isinstance(action_space, gym.spaces.Box):
    if len(action_space.shape) != 1:
      raise ValueError("when action space is an instance of gym.spaces.Box "
                       "it should have a single dimension, got "
                       f"len(action_space.shape) = {len(action_space.shape)}")
    output_units = [action_space.shape[0], *other_outputs]
    return MujocoModel(input_shape=observation_space.shape,
                       output_units=output_units)
  raise ValueError(f"unsupported action space {action_space}")
