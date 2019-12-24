""" Implements PPO Learner. """
from torch.optim import Adam
from derl.learners_torch.learner import Learner
from derl.models_torch import NatureDQN
from derl.policies_torch import ActorCriticPolicy
from derl.alg_torch.ppo import PPO
from derl.runners.onpolicy import make_ppo_runner
from derl.train_torch import StepVariable, linear_anneal


class PPOLearner(Learner):
  """ Proximal Policy Optimization learner. """

  @classmethod
  def get_parser_defaults(cls, env_type="atari"):
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
    return defaults.get(env_type)

  @staticmethod
  def make_runner(env, model=None, **kwargs):
    model = (model if model is not None
             else NatureDQN((env.action_space.n, 1)))
    policy = ActorCriticPolicy(model)
    runner_kwargs = {key: kwargs[key] for key in
                     ["gamma", "lambda_", "num_epochs", "num_minibatches"]
                     if key in kwargs}
    runner = make_ppo_runner(env, policy, kwargs["num_runner_steps"],
                             kwargs["num_train_steps"],
                             step_var=StepVariable.get_global_step(),
                             **runner_kwargs)
    return runner

  @staticmethod
  def make_alg(runner, **kwargs):
    lr = linear_anneal("lr", kwargs["lr"], kwargs["num_train_steps"],
                       step_var=runner.step_var)
    params = runner.policy.model.parameters()
    if "optimizer_epsilon" in kwargs:
      optimizer = Adam(params, lr, eps=kwargs["optimizer_epsilon"])
    else:
      optimizer = Adam(params, lr)

    ppo_kwargs = {key: kwargs[key]
                  for key in ["value_loss_coef", "entropy_coef",
                              "cliprange", "max_grad_norm"]
                  if key in kwargs}
    ppo = PPO(runner.policy, optimizer, **ppo_kwargs)
    return ppo
