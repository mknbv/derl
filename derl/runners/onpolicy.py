""" Environment runner wrappers. """
import numpy as np

from derl.runners.env_runner import EnvRunner, RunnerWrapper
from derl.runners.trajectory_transforms import (GAE,
                                                MergeTimeBatch,
                                                NormalizeAdvantages)
from derl.runners.summary import PeriodicSummaries


class TransformInteractions(RunnerWrapper):
  """ Transforms ineractions by applying a list of callables. """
  def __init__(self, runner, transforms=None, asarray=True):
    super().__init__(runner)
    self.transforms = transforms or []
    self.asarray = asarray

  def run(self, obs=None):
    for interactions in self.runner.run(obs=None):
      if self.asarray:
        for key, val in filter(lambda kv: kv[0] != "state",
                               interactions.items()):
          try:
            interactions[key] = np.asarray(val)
          except ValueError:
            raise ValueError(
                f"cannot convert value under key '{key}' to np.ndarray")
      for transform in self.transforms:
        transform(interactions)
      yield interactions


class IterateWithMinibatches(RunnerWrapper):
  """ Iterates over interactions with minibatches for a given number of epochs.
  """
  def __init__(self, runner, num_epochs=3, num_minibatches=4,
               shuffle_before_epoch=True):
    super().__init__(runner)
    self.num_epochs = num_epochs
    self.num_minibatches = num_minibatches
    self.shuffle_before_epoch = shuffle_before_epoch

  @staticmethod
  def shuffle_interactions(interactions):
    """ Shuffles given interactions. """
    sample_size = interactions["observations"].shape[0]
    indices = np.random.permutation(sample_size)
    for key, val in filter(lambda kv: kv[0] != "state", interactions.items()):
      interactions[key] = val[indices]

  def run(self, obs=None):
    for interactions in self.runner.run(obs=obs):
      for _ in range(self.num_epochs):
        if self.shuffle_before_epoch:
          IterateWithMinibatches.shuffle_interactions(interactions)

        sample_size = interactions["observations"].shape[0]
        mbsize = sample_size // self.num_minibatches
        for start in range(0, sample_size, mbsize):
          indices = np.arange(start, min(start + mbsize, sample_size))
          yield dict((key, val[indices]) if key != "state" else (key, val)
                     for key, val in interactions.items())


def ppo_runner_wrap(runner, gamma=0.99, lambda_=0.95,
                    num_epochs=3, num_minibatches=4):
  """ Wrapps given runner for PPO training. """
  env, policy = runner.env, runner.policy
  transforms = [GAE(policy, gamma=gamma, lambda_=lambda_, normalize=False)]
  if not policy.is_recurrent() and getattr(env.unwrapped, "nenvs", None):
    transforms.append(MergeTimeBatch())
  runner = TransformInteractions(runner, transforms)
  runner = IterateWithMinibatches(runner, num_epochs, num_minibatches)
  runner = TransformInteractions(runner, [NormalizeAdvantages()])
  return runner


def make_ppo_runner(env, policy, horizon, nsteps, nlogs=1e5, **wrap_kwargs):
  """ Creates and wraps env runner for PPO training. """
  runner = EnvRunner(env, policy, horizon, nsteps)
  runner = PeriodicSummaries.make_with_nlogs(runner, nlogs)
  return ppo_runner_wrap(runner, **wrap_kwargs)
