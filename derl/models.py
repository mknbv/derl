""" Keras models for reinforcement learning problems. """
from math import sqrt
import gym.spaces
import tensorflow as tf
from .common import clone_model
from .env.env_batch import SpaceBatch


def compute_outputs(activations, units, layer_type=tf.keras.layers.Dense,
                    **kwargs):
  """ Applies layers to a given tensor of activations. """
  if isinstance(units, (list, tuple)):
    return [layer_type(units=n, **kwargs)(activations) for n in units]
  return layer_type(units=units, **kwargs)(activations)


class BaseOutputsModel(tf.keras.Model):
  """ Adds output layers on top of the given base model. """
  def __init__(self, base, output_units, layer_type=tf.keras.layers.Dense,
               **kwargs):
    inputs = tf.keras.layers.Input(base.input.shape[1:])
    outputs = compute_outputs(base(inputs), output_units,
                              layer_type=layer_type, **kwargs)
    super().__init__(inputs=inputs, outputs=outputs)


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
               noisy=True,
               kernel_initializer=tf.initializers.orthogonal(sqrt(2)),
               bias_initializer=tf.initializers.zeros()):
    dense_class = NoisyDense if noisy else tf.keras.layers.Dense
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
        dense_class(
            units=512,
            activation=tf.nn.relu,
            kernel_initializer=kernel_initializer,
            bias_initializer=bias_initializer
        ),
    ])


class NoisyDense(tf.keras.layers.Layer):
  """ Adds noisy linear transformations to regular dense layer. """
  # pylint: disable=too-many-instance-attributes
  def __init__(self, units,
               stddev=0.5,
               factorized=True,
               activation=None,
               kernel_initializer="glorot_uniform",
               bias_initializer="zeros"):
    super().__init__()
    self.units = units
    self.stddev = stddev
    self.factorized = factorized
    self.activation = activation
    self.kernel_initializer = tf.keras.initializers.get(kernel_initializer)
    self.bias_initializer = tf.keras.initializers.get(bias_initializer)
    self.regular_kernel = None
    self.regular_bias = None
    self.noise_kernel = None
    self.noise_bias = None

  def build(self, input_shape):
    input_dim, output_dim = input_shape[-1].value, self.units
    self.regular_kernel = self.add_variable(
        "regular_kernel",
        shape=(input_dim, output_dim),
        initializer=self.kernel_initializer)
    self.regular_bias = self.add_variable(
        "regular_bias",
        shape=(output_dim,),
        initializer=self.bias_initializer)
    self.noise_kernel = self.add_variable(
        "noise_kernel",
        shape=(input_dim, output_dim),
        initializer=self.kernel_initializer)
    self.noise_bias = self.add_variable(
        "noise_bias",
        shape=(output_dim,),
        initializer=self.bias_initializer)

  def apply_dense(self, inputs, kernel, bias): # pylint: disable=no-self-use
    """ Applies dense layer with specified kernel and bias to given inputs. """
    return tf.nn.bias_add(tf.linalg.matmul(inputs, kernel), bias)

  def call(self, inputs): # pylint: disable=arguments-differ
    input_dim, output_dim = self.regular_kernel.shape.as_list()
    if self.factorized:
      input_noise = tf.random.normal(shape=(input_dim,), stddev=self.stddev)
      output_noise = tf.random.normal(shape=(output_dim,), stddev=self.stddev)
      kernel_noise = input_noise[:, None] * output_noise[None]
      bias_noise = output_noise
    else:
      kernel_noise = tf.random.normal(shape=(output_dim, input_dim),
                                      stddev=self.stddev)
      bias_noise = tf.random.normal(shape=(output_dim,), stddev=self.stddev)
    noisy_kernel = self.noise_kernel * kernel_noise
    noisy_bias = self.noise_bias * bias_noise

    regular_outputs = self.apply_dense(
        inputs, self.regular_kernel, self.regular_bias)
    noise_outputs = self.apply_dense(inputs, noisy_kernel, noisy_bias)
    outputs = tf.add(regular_outputs, noise_outputs)
    if self.activation is not None:
      outputs = self.activation(outputs)
    return outputs


class NatureDQNModel(tf.keras.Model):
  """ Nature DQN model with possibly several outputs. """
  # pylint: disable=too-many-arguments
  def __init__(self,
               output_units,
               input_shape=(84, 84, 4),
               ubyte_rescale=None,
               noisy=False,
               dueling=False,
               nbins=None,
               kernel_initializer=tf.initializers.orthogonal(sqrt(2)),
               bias_initializer=tf.initializers.zeros()):
    super().__init__()
    self.dueling = dueling
    self.nbins = nbins
    self.single_output = not isinstance(output_units, (list, tuple))
    self.output_units = ([output_units] if self.single_output
                         else list(output_units))
    if nbins is not None:
      self.output_units[0] *= nbins
    if dueling:
      self.output_units.append(nbins or 1)
    init = {"kernel_initializer": kernel_initializer,
            "bias_initializer": bias_initializer}
    self.base = NatureDQNBase(input_shape, ubyte_rescale, noisy=noisy, **init)
    dense_class = NoisyDense if noisy else tf.keras.layers.Dense
    self.output_layers = [dense_class(units=n, **init)
                          for n in self.output_units]

  @property
  def input(self):
    return self.base.input

  def call(self, inputs, training=True, mask=None):
    base_outputs = self.base(inputs)
    outputs = [layer(base_outputs) for layer in self.output_layers]
    if self.nbins is not None:
      nactions = self.output_units[0] // self.nbins
      outputs[0] = tf.reshape(outputs[0], (-1, nactions, self.nbins))
    if self.dueling:
      advantages, values = outputs[0], outputs.pop()
      values = tf.reshape(
          values, (-1, 1, self.nbins) if self.nbins is not None else (-1, 1))
      outputs[0] = (values + advantages
                    - tf.reduce_mean(advantages, 1, keepdims=True))
    if self.single_output:
      outputs = outputs[0]
    return outputs


class IMPALABase(tf.keras.Model):
  """ Hidden layers of IMPALA model.

  See [Espeholt et al.](https://arxiv.org/abs/1802.01561).
  """
  def __init__(self, input_shape=(84, 84, 4),
               ubyte_rescale=None,
               kernel_initializer=tf.initializers.orthogonal(sqrt(2)),
               bias_initializer=tf.initializers.zeros()):
    init = {"kernel_initializer": kernel_initializer,
            "bias_initializer": bias_initializer}

    def simple_block(filters, inputs, pool):
      out = tf.keras.layers.Conv2D(filters, kernel_size=3,
                                   padding="same", **init)(inputs)
      if pool:
        out = tf.keras.layers.MaxPool2D(pool_size=3, strides=2,
                                        padding="same")(out)
      return out

    inputs = tf.keras.layers.Input(input_shape)
    out = MaybeRescale(input_shape, ubyte_rescale)(inputs)
    for filters, nblocks in [(16, 2), (32, 2), (32, 2)]:
      out = simple_block(filters, out, True)

      for _ in range(nblocks):
        block_input = out
        out = tf.keras.layers.ReLU()(out)
        out = simple_block(filters, out, False)
        out = tf.keras.layers.ReLU()(out)
        out = simple_block(filters, out, False)
        out = tf.keras.layers.add([block_input, out])

    out = tf.keras.layers.ReLU()(out)
    out = tf.keras.layers.Flatten()(out)
    out = tf.keras.layers.Dense(units=256, activation=tf.nn.relu, **init)(out)

    # Instead of the LSTM layer we add an additional dense layer.
    out = tf.keras.layers.Dense(units=256, activation=tf.nn.relu, **init)(out)
    super().__init__(inputs=inputs, outputs=out)


class IMPALAModel(BaseOutputsModel):
  """ Non-recurrent version of the IMPALA model.

  Model from the paper [Espeholt et al.](https://arxiv.org/abs/1802.01561).
  """
  def __init__(self, output_units,
               input_shape=(84, 84, 4),
               ubyte_rescale=True,
               kernel_initializer=tf.initializers.orthogonal(sqrt(2)),
               bias_initializer=tf.initializers.zeros()):
    init = {"kernel_initializer": kernel_initializer,
            "bias_initializer": bias_initializer}
    super().__init__(IMPALABase(input_shape, ubyte_rescale, **init),
                     output_units, **init)


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
    model_outputs = self.model(inputs)
    if not isinstance(model_outputs, tf.Tensor):
      logits, *outputs = model_outputs
    else:
      logits, outputs = model_outputs, []

    std = tf.exp(self.logstd)
    std = tf.tile(std[None], [inputs.shape[0].value, 1])
    return (logits, std, *outputs)


def make_model(observation_space, action_space, other_outputs=None, **kwargs):
  """ Creates default model for given observation and action spaces. """
  if isinstance(other_outputs, int) or other_outputs is None:
    other_outputs = [other_outputs] if other_outputs is not None else []

  if isinstance(action_space, SpaceBatch):
    action_space = action_space.spaces[0]
  if isinstance(action_space, gym.spaces.Discrete):
    output_units = [action_space.n, *other_outputs]
    return NatureDQNModel(input_shape=observation_space.shape,
                          output_units=output_units, **kwargs)
  if isinstance(action_space, gym.spaces.Box):
    if len(action_space.shape) != 1:
      raise ValueError("when action space is an instance of gym.spaces.Box "
                       "it should have a single dimension, got "
                       f"len(action_space.shape) = {len(action_space.shape)}")
    output_units = [action_space.shape[0], *other_outputs]
    return MujocoModel(input_shape=observation_space.shape,
                       output_units=output_units, **kwargs)
  raise ValueError(f"unsupported action space {action_space}")
