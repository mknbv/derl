""" Advantage Actor-Critic Learner. """
from torch.optim import RMSprop
from derl.alg.a2c import A2C
from derl.alg.common import Trainer
from derl.anneal import LinearAnneal
from derl.learners.learner import Learner
from derl.models import make_model
from derl.policies import ActorCriticPolicy
from derl.runners.env_runner import EnvRunner
from derl.runners.onpolicy import TransformInteractions
from derl.runners.summary import PeriodicSummaries
from derl.runners.trajectory_transforms import GAE, MergeTimeBatch


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
  def make_runner(env, model=None, nlogs=1e5, **kwargs):
    if model is None:
      model = make_model(env.observation_space, env.action_space, 1)
    policy = ActorCriticPolicy(model)
    gae_kwargs = {k: kwargs[k] for k in ("gamma", "lambda_") if k in kwargs}
    gae_kwargs["normalize"] = kwargs.get("normalize_gae", False)
    runner = EnvRunner(env, policy, kwargs["num_runner_steps"],
                       nsteps=kwargs["num_train_steps"])
    runner = PeriodicSummaries.make_with_nlogs(runner, nlogs)
    transforms = [GAE(policy, **gae_kwargs)]
    if hasattr(env.unwrapped, "nenvs"):
      transforms.append(MergeTimeBatch())
    runner = TransformInteractions(runner, transforms)
    return runner


  @staticmethod
  def make_alg(runner, **kwargs):
    lr = LinearAnneal(kwargs["lr"], kwargs["num_train_steps"], 0., name="lr")
    optimizer_kwargs = {
        "alpha": kwargs.pop("optimizer_alpha", 0.99),
        "eps": kwargs.pop("optimizer_epsilon", 1e-5),
    }
    optimizer = RMSprop(runner.policy.model.parameters(),
                        lr.get_tensor(), **optimizer_kwargs)
    trainer = Trainer(optimizer, anneals=[lr],
                      max_grad_norm=kwargs.get("max_grad_norm"))
    a2c_kwargs = {k: kwargs[k] for k in ("value_loss_coef", "entropy_coef")
                  if k in kwargs}
    return A2C(runner, trainer, **a2c_kwargs)
