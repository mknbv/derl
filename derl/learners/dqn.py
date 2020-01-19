""" Implements Deep Q-Learning Learner. """
from copy import deepcopy
from torch.optim import RMSprop
from derl.alg.dqn import DQN
from derl.learners.learner import Learner
from derl.models import NatureCNNModel
from derl.policies import EpsilonGreedyPolicy
from derl.runners.experience_replay import make_dqn_runner
from derl.train import StepVariable


class DQNLearner(Learner):
  """ Deep Q-Learning Learner. """
  @classmethod
  def get_parser_defaults(cls, env_type="atari"):
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
    }.get(env_type)

  @staticmethod
  def make_model(env, init_fn=None, **kwargs):
    """ Creates Nature-DQN model for a given env. """
    return NatureCNNModel(input_shape=env.observation_space.shape,
                          output_units=env.action_space.n,
                          init_fn=init_fn, **kwargs)

  @staticmethod
  def make_runner(env, model=None, **kwargs):
    model = model or DQNLearner.make_model(
        env, noisy=kwargs.get("noisy", False),
        dueling=kwargs.get("dueling", True))
    step_var = StepVariable()
    epsilon = 0.
    if not kwargs.get("noisy", False):
      epsilon = step_var.linear_anneal(
          start_value=kwargs["exploration_epsilon_start"],
          nsteps=kwargs["exploration_end_step"],
          end_value=kwargs["exploration_epsilon_end"],
          name="exploration_epsilon")
    policy = EpsilonGreedyPolicy(model, epsilon)
    runner_kwargs = {k: kwargs[k] for k in ("storage_size", "storage_init_size",
                                            "batch_size", "steps_per_sample",
                                            "nstep", "prioritized")
                     if k in kwargs}
    runner = make_dqn_runner(env, policy, kwargs["num_train_steps"],
                             step_var=step_var, **runner_kwargs)
    return runner

  @staticmethod
  def make_alg(runner, **kwargs):
    model = runner.policy.model
    target_model = deepcopy(model)

    optimizer_kwargs = {
        "alpha": kwargs.get("decay", 0.95),
        "momentum": kwargs.get("momentum", 0.),
        "eps": kwargs.get("optimizer_epsilon", 0.01),
    }
    optimizer = RMSprop(model.parameters(), kwargs["lr"], **optimizer_kwargs)
    dqn_kwargs = {k: kwargs[k] for k in
                  ("gamma", "target_update_period", "double") if k in kwargs}
    alg = DQN(model, target_model, optimizer, **dqn_kwargs)
    return alg
