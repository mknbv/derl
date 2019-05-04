""" Implements PPO Learner. """
import tensorflow as tf
from derl.base import Learner
from derl.models import make_model
from derl.policies import ActorCriticPolicy
from derl.alg.ppo import PPO
from derl.runners import make_ppo_runner
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
    return defaults[env_type]

  @staticmethod
  def make_runner(env, args, model=None):
    model = (model if model is not None
             else make_model(env.observation_space, env.action_space, 1))
    policy = ActorCriticPolicy(model)
    runner = make_ppo_runner(env, policy, args.num_runner_steps,
                             gamma=args.gamma, lambda_=args.lambda_,
                             num_epochs=args.num_epochs,
                             num_minibatches=args.num_minibatches)
    return runner

  @staticmethod
  def make_alg(runner, args):
    lr = linear_anneal("lr", args.lr, args.num_train_steps,
                       step_var=runner.step_var)
    optimizer = tf.train.AdamOptimizer(lr, epsilon=args.optimizer_epsilon)
    ppo = PPO(runner.policy, optimizer,
              value_loss_coef=args.value_loss_coef,
              entropy_coef=args.entropy_coef,
              cliprange=args.cliprange,
              max_grad_norm=args.max_grad_norm)
    return ppo

  def learning_body(self):
    data = self.runner.get_next()
    loss = self.alg.step(data)
    yield data, loss
    while not self.runner.trajectory_is_stale():
      data = self.runner.get_next()
      loss = self.alg.step(data)
      yield data, loss
