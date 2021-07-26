""" PyTorch models for RL. """
from functools import partial, wraps
from itertools import chain, tee
from math import floor
import gym
import numpy as np
import torch
import torch.nn as nn
from derl.env.env_batch import SpaceBatch


class NoisyLinear(nn.Module):
  """ Noisy linear transformation on top of regular linear layer. """
  def __init__(self, in_features, out_features,
               stddev=0.5, factorized=True):
    super().__init__()
    self.linear = nn.Linear(in_features, out_features)
    self.stddev = stddev
    self.factorized = factorized
    self.weight = nn.Parameter(
        torch.Tensor(out_features, in_features))
    self.bias = nn.Parameter(torch.Tensor(out_features))
    self.reset_parameters()

  def reset_parameters(self):
    """ Reinitializes parameters of the layer. """
    self.linear.reset_parameters()
    nn.init.kaiming_uniform_(self.weight, a=np.sqrt(5))
    fan_in = self.weight.shape[1]
    bound = 1 / np.sqrt(fan_in)
    nn.init.uniform_(self.bias, -bound, bound)

  def sample_noise(self):
    """ Samples noise for forward method. """
    out_features, in_features = self.weight.shape
    if not self.factorized:
      weight_noise = torch.normal(0., self.stddev,
                                  size=(out_features, in_features))
      bias_noise = torch.normal(0., self.stddev, size=(out_features,))
      return weight_noise, bias_noise

    output_noise = torch.normal(0., self.stddev, size=(out_features,))
    input_noise = torch.normal(0., self.stddev, size=(in_features,))
    weight_noise = output_noise[:, None] * input_noise[None]
    bias_noise = output_noise
    return weight_noise, bias_noise

  def forward(self, inputs):  # pylint: disable=arguments-differ
    weight_noise, bias_noise = self.sample_noise()
    weight_noise = weight_noise.to(self.weight.device)
    bias_noise = bias_noise.to(self.bias.device)
    noisy_weight = self.weight * weight_noise
    noisy_bias = self.bias * bias_noise
    return (self.linear.forward(inputs)
            + nn.functional.linear(inputs, noisy_weight, noisy_bias))


def conv2d_output_shape(height, width, conv2d):
  """ Computes output shape of given conv2d layer. """
  padding, stride = conv2d.padding, conv2d.stride
  dilation, kernel_size = conv2d.dilation, conv2d.kernel_size
  out_height = floor(
      (height + 2 * padding[0] - dilation[0] * (kernel_size[0] - 1) - 1)
      / stride[0] + 1)
  out_width = floor(
      (width + 2 * padding[1] - dilation[1] * (kernel_size[1] - 1) - 1)
      / stride[1] + 1)
  return out_height, out_width


def collocate_inputs(device=True, dtype=True):
  """ Collocates inputs with model. """
  def decorator(forward):
    @wraps(forward)
    def wrapped(self, *inputs):
      collocated = []
      for inpt in inputs:
        if isinstance(inpt, np.ndarray):
          inpt = torch.from_numpy(inpt)
        model_device = next(self.parameters()).device
        model_dtype = next(self.parameters()).dtype
        if ((device and inpt.device != model_device)
            or (dtype and inpt.dtype != model_dtype)):
          target_device = model_device if device else None
          target_dtype = model_dtype if dtype else None
          inpt = inpt.to(device=target_device, dtype=target_dtype)
        collocated.append(inpt)
      return forward(self, *collocated)
    return wrapped
  return decorator


class NatureCNNBase(nn.Sequential):
  """ Hidden layers of the Nature DQN model. """
  def __init__(self, input_shape=(84, 84, 4), permute=True, noisy=False):
    super().__init__()
    self.permute = permute
    in_channels, height, width = input_shape
    if permute:
      height, width, in_channels = input_shape
    convolutions = [
        nn.Conv2d(in_channels, 32, 8, 4),
        nn.Conv2d(32, 64, 4, 2),
        nn.Conv2d(64, 64, 3, 1),
    ]
    for i, conv in enumerate(convolutions):
      height, width = conv2d_output_shape(height, width, conv)
      self.add_module(f"conv-{i}", conv)
      self.add_module(f"relu-{i}", nn.ReLU())

    self.add_module("flatten", nn.Flatten())
    in_features = height * width * convolutions[-1].out_channels
    linear_class = NoisyLinear if noisy else nn.Linear
    self.add_module("linear", linear_class(in_features, 512))

  @collocate_inputs(dtype=False)
  def forward(self, inputs):  # pylint: disable=arguments-differ
    if self.permute:
      inputs = inputs.permute(0, 3, 1, 2)
    if inputs.dtype == torch.uint8:
      inputs = inputs.float() / 255
    inputs = inputs.contiguous()
    return super().forward(inputs)


def init_weights(layer, weight_initializer, bias_initializer):
  """ Initializers weights and biases of a given layer. """
  if hasattr(layer, "weight"):
    weight_initializer(layer.weight)
  if hasattr(layer, "bias"):
    bias_initializer(layer.bias)


def orthogonal_init(layer):
  """ Orthogonal initialization of layers and zero initialization of biases. """
  return init_weights(layer,
                      weight_initializer=nn.init.orthogonal_,
                      bias_initializer=nn.init.zeros_)


def broadcast_inputs(ndims):
  """ Broadcast inputs to specified number of dims and then back. """
  def decorator(forward):
    @wraps(forward)
    def wrapped(self, *inputs):
      input_ndim = inputs[0].ndim
      for i, inpt in enumerate(inputs):
        if inpt.ndim != input_ndim:
          raise ValueError("for broadcasting all inputs must have the same "
                           "number of dimensions, got "
                           f"inputs[0].shape={inputs[0].shape}, "
                           f"inputs[{i}].shape={inputs[i].shape}")
      expand_dims = ndims - input_ndim
      inputs = tuple(inpt[(None,) * expand_dims] for inpt in inputs)
      outputs = forward(self, *inputs)

      def unbroadcast(output):
        if isinstance(output, (tuple, list)):
          return type(output)(unbroadcast(out) for out in output)
        return torch.reshape(output, output.shape[expand_dims:])
      return unbroadcast(outputs)
    return wrapped
  return decorator


class NatureCNNModel(nn.Module):
  """ Nature DQN model that supports subsequently proposed modifications. """
  def __init__(self,
               output_units,
               input_shape=(84, 84, 4),
               noisy=False,
               dueling=False,
               nbins=None,
               init_fn=orthogonal_init):
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

    self.base = NatureCNNBase(input_shape, noisy=noisy)
    in_units = list(self.base.children())[-1].out_features
    linear_class = NoisyLinear if noisy else nn.Linear
    self.output_layers = nn.ModuleList(
        [linear_class(in_units, out_units)
         for out_units in self.output_units])
    if init_fn:
      self.apply(init_fn)
    self.to("cuda" if torch.cuda.is_available() else "cpu")

  @broadcast_inputs(ndims=4)
  def forward(self, *inputs):
    observations, = inputs
    base_outputs = self.base(observations)
    outputs = [layer(base_outputs) for layer in self.output_layers]
    if self.nbins is not None:
      nactions = self.output_units[0] // self.nbins
      outputs[0] = torch.reshape(outputs[0], (-1, nactions, self.nbins))
    if self.dueling:
      advantages, values = outputs[0], outputs.pop()
      values = torch.reshape(
          values, (-1, 1, self.nbins) if self.nbins is not None else (-1, 1))
      outputs[0] = (values + advantages
                    - torch.mean(advantages, 1, keepdims=True))
    if self.single_output:
      outputs = outputs[0]
    return outputs


def pairwise(iterable):
  """ s -> (s0,s1), (s1,s2), (s2, s3), ... """
  it1, it2 = tee(iterable)
  next(it2, None)
  return zip(it1, it2)


class MLP(nn.Sequential):
  """ Multi-layer perceptron. """
  def __init__(self,
               in_features,
               out_features,
               hidden_features=(64, 64),
               activation=nn.Tanh):
    layers = list(chain.from_iterable(
        (nn.Linear(nin, nout), activation())
        for nin, nout in pairwise(chain(
            (in_features,), hidden_features, (out_features,)))
    ))
    layers.pop()  # Remove redundant activation after last layer.
    super().__init__(*layers)


class MuJoCoModel(nn.Module):
  """ MuJoCo model. """
  def __init__(self, observation_dim,
               output_units,
               mlp=MLP,
               init_fn=orthogonal_init):
    super().__init__()
    if not isinstance(output_units, (tuple, list)):
      output_units = [output_units]
    self.module_list = nn.ModuleList()
    for nunits in output_units:
      self.module_list.append(mlp(observation_dim, nunits))
    if init_fn is not None:
      self.apply(init_fn)
    self.logstd = nn.Parameter(torch.zeros(output_units[0]))
    self.to("cuda" if torch.cuda.is_available() else "cpu")

  @broadcast_inputs(ndims=2)
  @collocate_inputs()
  def forward(self, *inputs):
    observations, = inputs
    observations = observations.reshape((observations.shape[0], -1))
    logits, *outputs = (module(observations) for module in self.module_list)
    batch_size = observations.shape[0]
    std = torch.repeat_interleave(torch.exp(self.logstd)[None], batch_size, 0)
    return (logits, std, *outputs)


def vector_size(shape):
  """ Checks whether the given shape is 1-dim and returns its size. """
  if len(shape) != 1:
    raise ValueError(f"expected vector shape, got shape={shape}")
  return shape[0]


def make_model(observation_space, action_space, other_outputs=None, **kwargs):
  """ Creates default model for given observation and action spaces. """
  if isinstance(other_outputs, int) or other_outputs is None:
    other_outputs = [other_outputs] if other_outputs is not None else []

  if isinstance(action_space, SpaceBatch):
    action_space = action_space.spaces[0]
  if isinstance(action_space, gym.spaces.Discrete):
    output_units = [action_space.n, *other_outputs]
    return NatureCNNModel(input_shape=observation_space.shape,
                          output_units=output_units, **kwargs)
  if isinstance(action_space, gym.spaces.Box):
    observation_dim = vector_size(observation_space.shape)
    action_dim = vector_size(action_space.shape)
    output_units = [action_dim, *other_outputs]
    return MuJoCoModel(observation_dim=observation_dim,
                       output_units=output_units, **kwargs)
  raise ValueError(f"unsupported action space {action_space}")


class SACMLP(nn.Module):
  """ MLP for use with SAC model. """
  def __init__(self, in_features, out_features, nheads=2,
               hidden_features=(256, 256), activation=nn.ReLU):
    super().__init__()
    self.hidden = MLP(in_features=in_features,
                      out_features=hidden_features[-1],
                      hidden_features=hidden_features[:-1],
                      activation=activation)
    if nheads is not None and nheads < 1:
      raise ValueError("nheads must be either None or at least 1, "
                       f"got nheads={nheads}")
    self.nheads = nheads
    self.activation = activation()
    self.heads = nn.ModuleList(nn.Linear(hidden_features[-1], out_features)
                               for _ in range(nheads or 1))

  def forward(self, *inputs):
    hidden = self.activation(self.hidden.forward(*inputs))
    return [head(hidden) for head in self.heads][
        slice(None) if self.nheads is not None else 0]


class ContinuousQValueModel(nn.Module):
  """ Continuous Q-value model. """
  def __init__(self, observation_dim, action_dim,
               mlp=partial(SACMLP, nheads=None), init_fn=orthogonal_init):
    super().__init__()
    self.observation_dim = observation_dim
    self.action_dim = action_dim
    self.mlp = mlp(observation_dim + action_dim, 1)
    if init_fn is not None:
      self.apply(init_fn)
    self.to("cuda" if torch.cuda.is_available() else "cpu")

  @broadcast_inputs(ndims=2)
  @collocate_inputs()
  def forward(self, *inputs):
    observations, actions = inputs
    cat = torch.cat([observations, actions], -1)
    return self.mlp(cat)
