""" Wrapper for writing summaries. """
from collections import deque
from gym import Wrapper
import numpy as np
import tensorflow as tf


class Summaries(Wrapper):
  """ Writes env summaries."""
  def __init__(self, env, prefix=None, running_mean_size=100, step_var=None):
    super(Summaries, self).__init__(env)
    self.episode_counter = 0
    self.prefix = prefix or self.env.spec.id
    self.step_var = (step_var if step_var is not None
                     else tf.train.get_global_step())

    nenvs = getattr(self.env.unwrapped, "nenvs", 1)
    self.rewards = np.zeros(nenvs)
    self.had_ended_episodes = np.zeros(nenvs, dtype=np.bool)
    self.episode_lengths = np.zeros(nenvs)
    self.reward_queues = [deque([], maxlen=running_mean_size)
                          for _ in range(nenvs)]

  def should_write_summaries(self):
    """ Returns true if it's time to write summaries. """
    return np.all(self.had_ended_episodes)

  def add_summaries(self):
    """ Writes summaries. """
    tf.contrib.summary.scalar(
        f"{self.prefix}/total_reward",
        tf.reduce_mean([q[-1] for q in self.reward_queues]),
        step=self.step_var)
    tf.contrib.summary.scalar(
        f"{self.prefix}/reward_mean_{self.reward_queues[0].maxlen}",
        tf.reduce_mean([np.mean(q) for q in self.reward_queues]),
        step=self.step_var)
    tf.contrib.summary.scalar(
        f"{self.prefix}/episode_length",
        tf.reduce_mean(self.episode_lengths),
        step=self.step_var)
    if self.had_ended_episodes.size > 1:
      tf.contrib.summary.scalar(
          f"{self.prefix}/min_reward",
          min(q[-1] for q in self.reward_queues),
          step=self.step_var)
      tf.contrib.summary.scalar(
          f"{self.prefix}/max_reward",
          max(q[-1] for q in self.reward_queues),
          step=self.step_var)
    self.episode_lengths.fill(0)
    self.had_ended_episodes.fill(False)

  def step(self, action):
    obs, rew, done, info = self.env.step(action)
    self.rewards += rew
    self.episode_lengths[~self.had_ended_episodes] += 1

    info_collection = [info] if isinstance(info, dict) else info
    done_collection = [done] if isinstance(done, bool) else done
    done_indices = [i for i, info in enumerate(info_collection)
                    if info.get("real_done", done_collection[i])]
    for i in done_indices:
      if not self.had_ended_episodes[i]:
        self.had_ended_episodes[i] = True
      self.reward_queues[i].append(self.rewards[i])
      self.rewards[i] = 0

    if self.should_write_summaries():
      self.add_summaries()
    return obs, rew, done, info

  def reset(self, **kwargs):
    self.rewards.fill(0)
    self.episode_lengths.fill(0)
    self.had_ended_episodes.fill(False)
    return self.env.reset(**kwargs)
