""" Implements PPO Learner. """
import tensorflow as tf
from derl.base import Learner
from derl.models import make_model
from derl.policies import ActorCriticPolicy
from derl.alg.ppo import PPO
from derl.runners.online import make_ppo_runner
from derl.train import linear_anneal


class PPOLearner(Learner):
  """ Proximal Policy Optimization learner. """

  @staticmethod
  def get_defaults(env_type="atari"):
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
  def make_runner(env, args, model=None):
    model = (model if model is not None
             else make_model(env.observation_space, env.action_space, 1))
    policy = ActorCriticPolicy(model)
    kwargs = vars(args)
    runner_kwargs = {key: kwargs[key] for key in
                     ["gamma", "lambda_", "num_epochs", "num_minibatches"]
                     if key in kwargs}
    runner = make_ppo_runner(env, policy, args.num_runner_steps,
                             **runner_kwargs)
    return runner

  @staticmethod
  def make_alg(runner, args):
    lr = linear_anneal("lr", args.lr, args.num_train_steps,
                       step_var=runner.step_var)
    if hasattr(args, "optimizer_epsilon"):
      optimizer = tf.train.AdamOptimizer(lr, epsilon=args.optimizer_epsilon)
    else:
      optimizer = tf.train.AdamOptimizer(lr)

    kwargs = vars(args)
    ppo_kwargs = {key: kwargs[key]
                  for key in ["value_loss_coef", "entropy_coef",
                              "cliprange", "max_grad_norm"]
                  if key in kwargs}
    ppo = PPO(runner.policy, optimizer, **ppo_kwargs)
    return ppo

  def learning_body(self):
    data = self.runner.get_next()
    loss = self.alg.step(data)
    yield data, loss
    while not self.runner.trajectory_is_stale():
      data = self.runner.get_next()
      loss = self.alg.step(data)
      yield data, loss
