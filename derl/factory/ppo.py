""" Implements PPO factory. """
from torch.optim import Adam
from derl.alg.common import Trainer
from derl.anneal import LinearAnneal
from derl.factory.factory import Factory
from derl.models import make_model
from derl.policies import ActorCriticPolicy
from derl.alg.ppo import PPO
from derl.runners.onpolicy import make_ppo_runner


class PPOFactory(Factory):
  """ Proximal Policy Optimization learner. """
  def __init__(self, *, unused_kwargs=("nenvs",), **kwargs):
    super().__init__(unused_kwargs=unused_kwargs, **kwargs)

  @staticmethod
  def get_parser_defaults(args_type="atari"):
    defaults = {
        "atari": {
            "num-train-steps": 10e6,
            "nenvs": 8,
            "num-runner-steps": 128,
            "gamma": 0.99,
            "lambda_": 0.95,
            "num-epochs": 3,
            "num-minibatches": 4,
            "cliprange": 0.1,
            "value-loss-coef": 0.25,
            "entropy-coef": 0.01,
            "max-grad-norm": 0.5,
            "lr": 2.5e-4,
            "optimizer-epsilon": 1e-5,
        },
        "mujoco": {
            "num-train-steps": 1e6,
            "nenvs": dict(type=int, default=None),
            "num-runner-steps": 2048,
            "gamma": 0.99,
            "lambda_": 0.95,
            "num-epochs": 10,
            "num-minibatches": 32,
            "cliprange": 0.2,
            "value-loss-coef": 0.25,
            "entropy-coef": 0.,
            "max-grad-norm": 0.5,
            "lr": 3e-4,
            "optimizer-epsilon": 1e-5,
        }
    }
    return defaults.get(args_type)

  @classmethod
  def with_default_kwargs(cls, args_type="atari", unused_kwargs=("nenvs",),
                          **kwargs):
    return super().with_default_kwargs(args_type, unused_kwargs, **kwargs)

  @classmethod
  def from_args(cls, args_type="atari", unused_kwargs=("nenvs",), args=None):
    return super().from_args(args_type, unused_kwargs, args)

  def make_runner(self, env, nlogs=1e5, **kwargs):
    with self.custom_kwargs(**kwargs):
      model = (self.get_arg("model") if self.has_arg("model")
               else make_model(env.observation_space, env.action_space, 1))
      policy = ActorCriticPolicy(model)
      runner_kwargs = self.get_arg_dict("gamma", "lambda_",
                                        "num_epochs", "num_minibatches")
      runner = make_ppo_runner(env, policy, self.get_arg("num_runner_steps"),
                               self.get_arg("num_train_steps"), nlogs=nlogs,
                               **runner_kwargs)
      return runner

  def make_trainer(self, runner, **kwargs):
    with self.custom_kwargs(**kwargs):
      lr = LinearAnneal(*self.get_arg_list("lr", "num_train_steps"), name="lr")
      params = runner.policy.model.parameters()
      optimizer_kwargs = {"params": params, "lr": lr.get_tensor()}
      if self.has_arg("optimizer_epsilon"):
        optimizer_kwargs["eps"] = self.get_arg("optimizer_epsilon")
      optimizer = Adam(**optimizer_kwargs)
      return Trainer(optimizer, anneals=[lr],
                     max_grad_norm=self.get_arg("max_grad_norm"))

  def make_alg(self, runner, trainer, **kwargs):
    with self.custom_kwargs(**kwargs):
      ppo_kwargs = self.get_arg_dict("value_loss_coef",
                                     "entropy_coef",
                                     "cliprange")
      ppo = PPO(runner, trainer, **ppo_kwargs)
      return ppo
