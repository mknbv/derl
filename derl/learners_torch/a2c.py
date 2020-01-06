""" Advantage Actor-Critic Learner. """
from torch.optim import RMSprop
from derl.alg_torch.a2c import A2C
from derl.learners_torch.learner import Learner
from derl.models_torch import make_module
from derl.policies_torch import ActorCriticPolicy
from derl.runners.env_runner import EnvRunner
from derl.runners.onpolicy import TransformInteractions
from derl.runners.trajectory_transforms import GAE, MergeTimeBatch
from derl.train_torch import StepVariable, linear_anneal


class A2CLearner(Learner):
  """ Advantage Actor-Critic Learner. """
  @classmethod
  def get_parser_defaults(cls, env_type="atari"):
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
    }.get(env_type)

  @staticmethod
  def make_runner(env, model=None, **kwargs):
    if model is None:
      model = make_module(env.observation_space, env.action_space, 1)
    policy = ActorCriticPolicy(model)
    gae_kwargs = {k: kwargs[k] for k in ("gamma", "lambda_") if k in kwargs}
    if "normalize_gae" in kwargs:
      gae_kwargs["normalize"] = kwargs.get("normalize_gae")
    runner = EnvRunner(env, policy, kwargs["num_runner_steps"],
                       nsteps=kwargs["num_train_steps"],
                       step_var=StepVariable.get_global_step())
    runner = TransformInteractions(
        runner, [GAE(policy, **gae_kwargs), MergeTimeBatch()])
    return runner


  @staticmethod
  def make_alg(runner, **kwargs):
    lr = linear_anneal("lr", kwargs["lr"], kwargs["num_train_steps"],
                       runner.step_var)
    optimizer_kwargs = {
        "alpha": kwargs.pop("optimizer_alpha", 0.99),
        "eps": kwargs.pop("optimizer_epsilon", 1e-5),
    }
    optimizer = RMSprop(runner.policy.model.parameters(),
                        lr, **optimizer_kwargs)
    a2c_kwargs = {k: kwargs[k] for k in
                  ("value_loss_coef", "entropy_coef", "max_grad_norm")
                  if k in kwargs}
    return A2C(runner.policy, optimizer, **a2c_kwargs)
