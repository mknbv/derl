""" QR-DQN Learner. """
import tensorflow as tf
from derl.alg.qr_dqn import QR_DQN
from derl.learners.dqn import DQNLearner
from derl.runners.experience_replay import make_dqn_runner
from derl.policies import EpsilonGreedyPolicy
from derl.train import StepVariable, linear_anneal


class QR_DQNLearner(DQNLearner): # pylint: disable=invalid-name
  """ Distributional RL with quantile regression learner. """
  @classmethod
  def get_parser_defaults(cls, env_type="atari"):
    defaults = DQNLearner.get_parser_defaults(env_type)
    if defaults is None:
      return defaults
    defaults["nbins"] = 200
    defaults["lr"] = 5e-5
    defaults["optimizer-epsilon"] = 0.01 / 32
    defaults["no-huber-loss"] = dict(action="store_false", dest="huber_loss")
    return defaults

  @staticmethod
  def make_runner(env, args, model=None):
    model = DQNLearner.make_model(env, nbins=vars(args).get("nbins", 200))
    step_var = StepVariable("global_step", tf.train.create_global_step())
    epsilon = linear_anneal(
        "exploration_epsilon", args.exploration_epsilon_start,
        args.exploration_end_step, step_var, args.exploration_epsilon_end)
    policy = EpsilonGreedyPolicy(model, epsilon)
    kwargs = vars(args)
    runner_kwargs = {k: kwargs[k] for k in ("storage_size", "storage_init_size",
                                            "batch_size", "steps_per_sample",
                                            "nstep")
                     if k in kwargs}
    runner = make_dqn_runner(env, policy, args.num_train_steps,
                             step_var=step_var, **runner_kwargs)
    return runner

  @staticmethod
  def make_alg(runner, args):
    model = runner.policy.model
    target_model = DQNLearner.make_model(
        runner.env, nbins=model.output.shape[-1].value)
    target_model.set_weights(model.get_weights())
    epsilon = vars(args).get("optimizer_epsilon", 0.01 / 32)
    optimizer = tf.train.AdamOptimizer(args.lr, epsilon=epsilon)

    kwargs = vars(args)
    alg_kwargs = {k: kwargs[k] for k in ("target_update_period",
                                         "double", "gamma", "huber_loss")
                  if k in kwargs}
    alg = QR_DQN(model, target_model, optimizer, **alg_kwargs)
    return alg
