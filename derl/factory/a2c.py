""" Advantage Actor-Critic Learner. """
from torch.optim import RMSprop
from derl.alg.a2c import A2C
from derl.alg.common import Trainer
from derl.anneal import LinearAnneal
from derl.factory.factory import Factory
from derl.models import make_model
from derl.policies import ActorCriticPolicy
from derl.runners.env_runner import EnvRunner
from derl.runners.onpolicy import TransformInteractions
from derl.runners.summary import PeriodicSummaries
from derl.runners.trajectory_transforms import GAE, MergeTimeBatch


class A2CFactory(Factory):
  """ Advantage Actor-Critic Learner. """
  @staticmethod
  def get_parser_defaults(args_type="atari"):
    return {
        "atari": {
            "nenvs": 8,
            "num-train-steps": 10e6,
            "num-runner-steps": 5,
            "gamma": 0.99,
            "lambda_": 1.,
            "normalize-gae": dict(action="store_true"),
            "lr": 7e-4,
            "optimizer-alpha": 0.99,
            "optimizer-epsilon": 1e-5,
            "value-loss-coef": 0.5,
            "entropy-coef": 0.01,
            "max-grad-norm": 0.5,
        }
    }.get(args_type)

  def make_runner(self, env, nlogs=1e5, **kwargs):
    with self.custom_kwargs(**kwargs):
      model = (self.get_arg("model") if self.has_arg("model")
               else make_model(env.observation_space, env.action_space, 1))
      policy = ActorCriticPolicy(model)
      gae_kwargs = self.get_arg_dict("gamma", "lambda_")
      gae_kwargs["normalize"] = self.get_arg_default("normalize_gae", False)
      runner = EnvRunner(env, policy, self.get_arg("num_runner_steps"),
                         nsteps=self.get_arg("num_train_steps"))
      runner = PeriodicSummaries.make_with_nlogs(runner, nlogs)
      transforms = [GAE(policy, **gae_kwargs)]
      if hasattr(env.unwrapped, "nenvs"):
        transforms.append(MergeTimeBatch())
      runner = TransformInteractions(runner, transforms)
      return runner

  def make_trainer(self, runner, **kwargs):
    with self.custom_kwargs(**kwargs):
      lr = LinearAnneal(self.get_arg("lr"), self.get_arg("num_train_steps"),
                        0., name="lr")
      optimizer_kwargs = {
          "alpha": self.get_arg_default("optimizer_alpha", 0.99),
          "eps": self.get_arg_default("optimizer_epsilon", 1e-55)
      }
      optimizer = RMSprop(runner.policy.model.parameters(),
                          lr.get_tensor(), **optimizer_kwargs)
      return Trainer(optimizer, anneals=[lr],
                     max_grad_norm=self.get_arg_default("max_grad_norm"))

  def make_alg(self, runner, trainer, **kwargs):
    with self.custom_kwargs(**kwargs):
      a2c_kwargs = self.get_arg_dict("value_loss_coef", "entropy_coef")
      return A2C(runner, trainer, **a2c_kwargs)
