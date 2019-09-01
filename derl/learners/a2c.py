""" Defines Advantage Actor-Critic Learner. """
import tensorflow as tf
from derl.alg.a2c import A2C
from derl.learners import Learner
from derl.models import make_model
from derl.policies import ActorCriticPolicy
from derl.runners.online import EnvRunner
from derl.runners.trajectory_transforms import GAE, MergeTimeBatch
from derl.train import linear_anneal


class A2CLearner(Learner):
  """ Advantage Actor-Critic Learner. """
  @staticmethod
  def get_defaults(env_type="atari"):
    return {
        "atari": {
            "nenvs": 8,
            "num-train-steps": 10e6,
            "num-runner-steps": 5,
            "gamma": 0.99,
            "lambda_": 1.,
            "normalize-gae": dict(action="store_true"),
            "lr": 7e-4,
            "optimizer-decay": 0.99,
            "optimizer-epsilon": 1e-5,
            "value-loss-coef": 0.5,
            "entropy-coef": 0.01,
            "max-grad-norm": 0.5,
        }
    }.get(env_type)

  @staticmethod
  def make_runner(env, args, model=None):
    if model is None:
      model = make_model(env.observation_space, env.action_space, 1)
    policy = ActorCriticPolicy(model)
    kwargs = vars(args)
    gae_kwargs = {k: kwargs[k] for k in ("gamma", "lambda_") if k in kwargs}
    if "normalize_gae" in kwargs:
      gae_kwargs["normalize"] = kwargs.get("normalize_gae")
    transforms = [GAE(policy, **gae_kwargs), MergeTimeBatch()]
    return EnvRunner(env, policy, args.num_runner_steps, transforms=transforms)

  @staticmethod
  def make_alg(runner, args):
    lr = linear_anneal("lr", args.lr, args.num_train_steps, runner.step_var)
    kwargs = vars(args)
    optimizer_kwargs = {
        "decay": kwargs.pop("optimizer_decay", 0.99),
        "epsilon": kwargs.pop("optimizer_epsilon", 1e-5),
    }
    optimizer = tf.train.RMSPropOptimizer(lr, **optimizer_kwargs)
    a2c_kwargs = {k: kwargs[k] for k in
                  ("value_loss_coef", "entropy_coef", "max_grad_norm")
                  if k in kwargs}
    return A2C(runner.policy, optimizer, **a2c_kwargs)
