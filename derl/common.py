""" Code common for the whole package. """
import tensorflow as tf


def flat_layers_iterator(model):
  """ Iterator over all layers of a given model. """
  layers = list(reversed(model.layers))
  while layers:
    toplayer = layers.pop()
    if hasattr(toplayer, "layers"):
      layers.extend(reversed(toplayer.layers))
    else:
      yield toplayer


def clone_layer(layer):
  """ Clones a given layer. """
  return layer.__class__.from_config(layer.get_config())


def clone_model(model, name=None):
  """ Clones a sequential model. """
  if not isinstance(model, tf.keras.Sequential):
    raise ValueError("can only copy models of type Sequential, got model "
                     f"type {type(model)}")

  def _is_int(s):
    # pylint: disable=invalid-name
    if s and s[0] in ('-', '+'):
      return s[1:].isdigit()
    return s.isdigit()

  if name is None:
    *name_parts, ending = model.name.split('_')
    if _is_int(ending):
      ending = int(ending) + 1
      name_parts.append(ending)
      name = '_'.join(name_parts)
    else:
      name_parts.append(ending)
      name_parts.append('copy')
      name = '_'.join(name_parts)

  # Use model._layers to ensure that all layers are cloned. The model's layers
  # property will exclude the initial InputLayer (if it exists) in the model,
  # resulting in a different Sequential model structure.
  # pylint: disable=protected-access
  layers = [clone_layer(layer) for layer in model._layers]
  return tf.keras.Sequential(layers, name=name)


def r_squared(targets, predictions):
  """ Computes coefficient of determination. """
  targets, predictions = map(tf.convert_to_tensor, (targets, predictions))
  target_variance = tf.nn.moments(targets, tuple(range(targets.ndim)))[1]
  return 1. - tf.reduce_mean(tf.square(targets - predictions)) / target_variance


def reduce_add_summary(name, tensor, family=None, step=None,
                       reduction=tf.reduce_mean):
  """ Reduces tensor with given function and adds resulting scalar summary."""
  reduced = reduction(tensor) if reduction else tensor
  if reduced.ndim != 0:
    raise ValueError("the result of applying reduction is not scalar and has "
                     f"shape {reduced.shape}")
  tf.contrib.summary.scalar(name, reduced, family=family, step=step)


# pylint: disable=invalid-name
def maybe_clip_by_global_norm_with_summary(summary_tag, tensors,
                                           clip_norm=None,
                                           step=None):
  """ Clips by clip_norm if specified, always adds summary of true norm. """
  if clip_norm is None:
    grad_norm = tf.linalg.global_norm(tensors)
  else:
    tensors, grad_norm = tf.clip_by_global_norm(tensors, clip_norm)
  tf.contrib.summary.scalar(summary_tag, grad_norm, step=step)
  return tensors
