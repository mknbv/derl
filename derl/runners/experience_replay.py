""" Implements experience replay. """
from derl.runners.env_runner import EnvRunner, RunnerWrapper
from derl.runners.onpolicy import TransformInteractions
from derl.runners.storage import InteractionStorage


class ExperienceReplay(RunnerWrapper):
  """ Saves interactions to storage and samples from it. """
  def __init__(self, runner, storage, storage_init_size=50_000,
               batch_size=32, nstep=3):
    super().__init__(runner)
    self.storage = storage
    self.storage_init_size = storage_init_size
    self.batch_size = batch_size
    self.nstep = nstep

  def initialize_storage(self):
    """ Initializes the storage with random interactions with environment. """
    if self.storage.size != 0:
      raise ValueError(f"Storage has size {self.storage.size}, but "
                       "but initialization requires it to be empty")
    obs = self.env.reset()
    for _ in range(self.storage_init_size):
      action = self.env.action_space.sample()
      next_obs, rew, done, _ = self.env.step(action)
      self.storage.add(obs, action, rew, done)
      obs = next_obs if not done else self.env.reset()
    return obs

  def run(self, obs=None):
    if obs is not None:
      raise ValueError("obs can only be None when running with experience "
                       f"replay, got {obs}")
    obs = self.initialize_storage()
    for interactions in self.runner.run(obs=obs):
      interactions = [interactions[k] for k in ("observations", "actions",
                                                "rewards", "resets")]
      self.storage.add_batch(*interactions)
      yield self.storage.sample(self.batch_size, self.nstep)


def dqn_runner_wrap(runner, storage_size=1_000_000, storage_init_size=50_000,
                    batch_size=32, nstep=3):
  """ Wraps runner as it is typically used with DQN alg. """
  storage = InteractionStorage(storage_size)
  return ExperienceReplay(runner, storage, storage_init_size,
                          batch_size, nstep)


def make_dqn_runner(env, policy, num_train_steps, steps_per_sample=4,
                    step_var=None, **wrap_kwargs):
  """ Creates experience replay runner as used typically used with DQN alg. """
  runner = EnvRunner(env, policy, horizon=steps_per_sample,
                     nsteps=num_train_steps, step_var=step_var)
  runner = TransformInteractions(runner)
  return dqn_runner_wrap(runner, **wrap_kwargs)
