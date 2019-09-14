""" Implements experience replay. """
from derl.runners.env_runner import EnvRunner, RunnerWrapper
from derl.runners.onpolicy import TransformInteractions
from derl.runners.storage import InteractionStorage


class ExperienceReplay(RunnerWrapper):
  """ Saves interactions to experience replay and samples from it. """
  def __init__(self, runner, storage, batch_size, nstep=3):
    super().__init__(runner)
    self.storage = storage
    self.batch_size = batch_size
    self.nstep = nstep

  def __iter__(self):
    for interactions in self.runner:
      interactions = [interactions[k] for k in ("observations", "actions",
                                                "rewards", "resets")]
      self.storage.add_batch(*interactions)
      yield self.storage.sample(self.batch_size, self.nstep)


def dqn_runner_wrap(runner, storage_size=1_000_000, batch_size=32, nstep=3,
                    init_size=50_000):
  """ Wraps runner as it is typically used with DQN alg. """
  storage = InteractionStorage.from_env(runner.env, storage_size, init_size)
  return ExperienceReplay(runner, storage, batch_size, nstep)


def make_dqn_runner(env, policy, num_train_steps, steps_per_sample=4,
                    step_var=None, **wrap_kwargs):
  """ Creates experience replay runner as used typically used with DQN alg. """
  runner = EnvRunner(env, policy, horizon=steps_per_sample,
                     nsteps=num_train_steps, step_var=step_var)
  runner = TransformInteractions(runner)
  return dqn_runner_wrap(runner, **wrap_kwargs)
