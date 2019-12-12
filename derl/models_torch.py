""" PyTorch models for RL. """
from math import floor
import numpy as np
import torch
import torch.nn as nn


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


class NatureDQNBase(nn.Sequential):
  """ Hidden layers of the Nature DQN model. """
  def __init__(self, input_shape=(84, 84, 4), permute=True):
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
    self.add_module("linear", nn.Linear(in_features, 512))

  def forward(self, inputs):  # pylint: disable=arguments-differ
    if isinstance(inputs, np.ndarray):
      inputs = torch.from_numpy(inputs)
    device = next(self.parameters()).device
    if inputs.device != device:
      inputs = inputs.to(device)
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
    def wrapped(self, inputs):
      expand_dims = ndims - len(inputs.shape)
      inputs = inputs[(None,) * expand_dims]
      outputs = forward(self, inputs)
      if expand_dims:
        if isinstance(outputs, (tuple, list)):
          return type(outputs)(
              map(lambda t: torch.reshape(t, t.shape[expand_dims:]),
                  outputs)
          )
        return torch.reshape(outputs, outputs.shape[expand_dims:])
      return outputs
    return wrapped
  return decorator


class NatureDQN(nn.Module):
  """ Nature DQN model that supports subsequently proposed modifications. """
  def __init__(self,
               output_units,
               input_shape=(84, 84, 4),
               dueling=False,
               nbins=None,
               qvalues_from_outputs=None,
               init_fn=orthogonal_init):
    super().__init__()
    self.dueling = dueling
    self.nbins = nbins
    if not qvalues_from_outputs:
      qvalues_from_outputs = lambda outputs: outputs
    self.qvalues_from_outputs = qvalues_from_outputs
    self.single_output = not isinstance(output_units, (list, tuple))
    self.output_units = ([output_units] if self.single_output
                         else list(output_units))

    if nbins is not None:
      self.output_units[0] *= nbins
    if dueling:
      self.output_units.append(nbins or 1)

    self.base = NatureDQNBase(input_shape)
    in_units = list(self.base.children())[-1].out_features
    self.output_layers = nn.ModuleList(
        [nn.Linear(in_units, out_units)
         for out_units in self.output_units])
    if init_fn:
      self.apply(init_fn)

  @classmethod
  def categorical(cls, output_units,
                  input_shape=(84, 84, 4),
                  dueling=False,
                  nbins=None,
                  val_range=(-10., 10.),
                  init_fn=orthogonal_init):
    """ Categorical distributional model. """
    if nbins is None:
      raise ValueError("nbins cannot be None")
    def qvalues_from_outputs(preds):
      values = torch.linspace(*val_range, nbins)
      return torch.sum(values * preds, -1)
    return cls(output_units,
               input_shape=input_shape,
               dueling=dueling,
               nbins=nbins,
               qvalues_from_outputs=qvalues_from_outputs,
               init_fn=init_fn)

  @classmethod
  def quantile(cls, output_units,
               input_shape=(84, 84, 4),
               dueling=False,
               nbins=None,
               init_fn=orthogonal_init):
    """ Quantile distributional model. """
    if nbins is None:
      raise ValueError("nbins cannot be None")
    def qvalues_from_outputs(preds):
      return torch.mean(preds, -1)
    return cls(output_units,
               input_shape=input_shape,
               dueling=dueling,
               nbins=nbins,
               qvalues_from_outputs=qvalues_from_outputs,
               init_fn=init_fn)

  @broadcast_inputs(ndims=4)
  def forward(self, inputs):  # pylint: disable=arguments-differ
    base_outputs = self.base(inputs)
    outputs = [layer(base_outputs) for layer in self.output_layers]
    if self.nbins is not None:
      nactions = self.output_units[0] // self.nbins
      outputs[0] = torch.reshape(outputs[0], (-1, nactions, self.nbins))
    if self.dueling:
      advantages, values = outputs[0], outputs.pop()
      values = torch.reshape(values,
                             (-1, 1, self.nbins) if self.nbins is not None
                             else (-1, 1))
      outputs[0] = (values + advantages
                    - torch.mean(advantages, 1, keepdims=True))
    if self.single_output:
      outputs = outputs[0]
    return outputs
