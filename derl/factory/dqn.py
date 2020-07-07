""" Implements Deep Q-Learning Learner. """
from torch.optim import RMSprop
from derl.alg.common import Trainer
from derl.alg.dqn import DQN
from derl.anneal import LinearAnneal
from derl.factory.factory import Factory
from derl.models import NatureCNNModel
from derl.policies import EpsilonGreedyPolicy
from derl.runners.experience_replay import make_dqn_runner


class DQNFactory(Factory):
  """ Deep Q-Learning Learner. """
  @staticmethod
  def get_parser_defaults(args_type="atari"):
    return {
        "atari": {
            "num-train-steps": 200e6,
            "no-dueling": dict(action="store_false", dest="dueling"),
            "noisy": dict(action="store_true"),
            "exploration-epsilon-start": 1.,
            "exploration-epsilon-end": 0.01,
            "exploration-end-step": int(1e6),
            "storage-size": int(1e6),
            "storage-init-size": int(50e3),
            "not-prioritized": dict(action="store_false", dest="prioritized"),
            "per-alpha": 0.6,
            "per-beta": dict(type=float, default=(0.4, 1.), nargs=2),
            "steps-per-sample": 4,
            "batch-size": 32,
            "nstep": 3,
            "lr": 2.5e-4,
            "optimizer-decay": 0.95,
            "optimizer-momentum": 0.,
            "optimizer-epsilon": 0.01,
            "gamma": 0.99,
            "target-update-period": int(10e3),
            "no-double": dict(action="store_false", dest="double"),
        },
    }.get(args_type)

  def make_model(self, env, init_fn=None, **kwargs):
    """ Creates Nature-DQN model for a given env. """
    with self.custom_kwargs(**kwargs):
      model_kwargs = self.get_arg_dict("noisy", "dueling", "nbins")
      return NatureCNNModel(input_shape=env.observation_space.shape,
                            output_units=env.action_space.n,
                            **model_kwargs,
                            init_fn=init_fn)

  def make_runner(self, env, nlogs=1e5, **kwargs):
    with self.custom_kwargs():
      noisy = self.get_arg_default("noisy", False)
      model = (self.get_arg("model") if self.has_arg("model") else
               self.make_model(env, noisy=noisy,
                               dueling=self.get_arg_default("dueling", True)))
      epsilon = 0.
      anneals = []
      if not noisy:
        epsilon_anneal = LinearAnneal(
            start=self.get_arg("exploration_epsilon_start"),
            nsteps=self.get_arg("exploration_end_step"),
            end=self.get_arg("exploration_epsilon_end"),
            name="exploration_epsilon")
        epsilon = epsilon_anneal.get_tensor()
        anneals.append(epsilon_anneal)
      policy = EpsilonGreedyPolicy(model, epsilon)
      runner_kwargs = self.get_arg_dict("storage_size", "storage_init_size",
                                        "batch_size", "steps_per_sample",
                                        "nstep", "prioritized")
      if self.has_arg("per_alpha"):
        runner_kwargs["alpha"] = self.get_arg("per_alpha")
      if self.has_arg("per_beta"):
        runner_kwargs["beta"] = self.get_arg("per_beta")
      runner = make_dqn_runner(env, policy, self.get_arg("num_train_steps"),
                               anneals=anneals, nlogs=nlogs, **runner_kwargs)
      return runner

  def make_trainer(self, runner, **kwargs):
    with self.custom_kwargs(**kwargs):
      model = runner.policy.model
      optimizer_kwargs = {
          "alpha": self.get_arg_default("optimizer_decay", 0.95),
          "momentum": self.get_arg_default("optimizer_momentum", 0.),
          "eps": self.get_arg_default("optimizer_epsilon", 0.01)
      }
      optimizer = RMSprop(model.parameters(), self.get_arg("lr"),
                          **optimizer_kwargs)
      return Trainer(optimizer)

  def make_alg(self, runner, trainer, **kwargs):
    with self.custom_kwargs():
      dqn_kwargs = self.get_arg_dict("gamma", "target_update_period", "double")
      alg = DQN(runner, trainer, **dqn_kwargs)
      return alg
