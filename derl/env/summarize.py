""" Wrapper for writing summaries. """
from collections import deque
from gym import Wrapper
import numpy as np
import tensorflow as tf


class RewardSummarizer:
  """ Summarizes rewards received from environment. """
  def __init__(self, nenvs, prefix, running_mean_size=100, step_var=None):
    self.prefix = prefix
    self.step_var = step_var
    self.had_ended_episodes = np.zeros(nenvs, dtype=np.bool)
    self.rewards = np.zeros(nenvs)
    self.episode_lengths = np.zeros(nenvs)
    self.reward_queues = [deque([], maxlen=running_mean_size)
                          for _ in range(nenvs)]

  def should_add_summaries(self):
    """ Returns `True` if it is time to write summaries. """
    return (
        tf.contrib.summary.should_record_summaries()
        and np.all(self.had_ended_episodes))

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

  def step(self, rewards, resets):
    """ Takes statistics from last env step and tries to add summaries.  """
    self.rewards += rewards
    self.episode_lengths[~self.had_ended_episodes] += 1
    for i, in zip(*resets.nonzero()):
      self.reward_queues[i].append(self.rewards[i])
      self.rewards[i] = 0
      self.had_ended_episodes[i] = True

    if self.should_add_summaries():
      self.add_summaries()
      self.episode_lengths.fill(0)
      self.had_ended_episodes.fill(False)


class Summarize(Wrapper):
  """ Writes env summaries."""
  def __init__(self, env, summarizer):
    super(Summarize, self).__init__(env)
    self.summarizer = summarizer

  @classmethod
  def reward_summarizer(cls, env, prefix=None, running_mean_size=100,
                        step_var=None):
    """ Creates an instance with reward summarizer. """
    nenvs = getattr(env.unwrapped, "nenvs", 1)
    prefix = prefix if prefix is not None else env.spec.id
    summarizer = RewardSummarizer(nenvs, prefix,
                                  running_mean_size=running_mean_size,
                                  step_var=step_var)
    return cls(env, summarizer)

  def step(self, action):
    obs, rew, done, info = self.env.step(action)

    info_collection = [info] if isinstance(info, dict) else info
    done_collection = [done] if isinstance(done, bool) else done
    resets = np.asarray([info.get("real_done", done_collection[i])
                         for i, info in enumerate(info_collection)])
    self.summarizer.step(rew, resets)

    return obs, rew, done, info

  def reset(self, **kwargs):
    return self.env.reset(**kwargs)
