""" Defines rainbow learner. """
from derl.learners.learner import Learner
from derl.learners.qr_dqn import QR_DQNLearner


class RainbowLearner(Learner):
  """ Rainbow learner. """
  @classmethod
  def get_parser_defaults(cls, env_type="atari"):
    return {
        "atari": {
            "num-train-steps": int(200e6),
            "distributional": dict(choices=["qr-dqn-1", "qr-dqn-0"],
                                   default="qr-dqn-1"),
            "nbins": 200,
            "no-noisy": dict(action="store_false", dest="noisy"),
            "noisy-stddev": 0.5,
            "no-dueling": dict(action="store_false", dest="dueling"),
            "exploration-epsilon-start": 0.,
            "exploration-epsilon-end": 0.,
            "exploration-end-step": int(1e6),
            "storage-size": int(1e6),
            "storage-init-size": int(20e3),
            "no-prioritized": dict(action="store_false", dest="prioritized"),
            "per-alpha": 0.5,
            "per-beta": dict(type=float, default=(0.4, 1.), nargs=2),
            "steps-per-sample": 4,
            "batch-size": 32,
            "nstep": 3,
            "lr": 6.25e-5,
            "optimizer-epsilon": 1.5e-4,
            "gamma": .99,
            "target-update-period": int(8e3),
            "no-double": dict(action="store_false", dest="double"),
        }
    }.get(env_type)

  @staticmethod
  def make_runner(env, model=None, **kwargs):
    return QR_DQNLearner.make_runner(env, model=model, **kwargs)

  @staticmethod
  def make_alg(runner, **kwargs):
    kwargs["no_huber_loss"] = kwargs.pop("distributional", None) == "qr-dqn-0"
    return QR_DQNLearner.make_alg(runner, **kwargs)
