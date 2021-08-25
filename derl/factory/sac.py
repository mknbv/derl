""" Implements SAC factory. """
import torch
from torch.optim import Adam
from derl.factory.factory import Factory
from derl.models import  SACModel
from derl.policies import SACPolicy
from derl.alg.sac import SAC, SACTrainer
from derl.runners.experience_replay import make_sac_runner


class SACFactory(Factory):
  """ Soft Actor-Critic factory. """

  @staticmethod
  def get_parser_defaults(args_type="atari"):
    return {
        "mujoco": {
            "num-train-steps": 1e6,
            "storage-size": int(1e6),
            "storage-init-size": 1000,
            "batch-size": 256,
            "steps-per-sample": 1000,
            "num-storage-samples": 1000,
            "reward-scale": 1.,
            "gamma": 0.99,
            "target-update-period": 1,
            "target-update-coef": 0.005,
            "policy-lr": 3e-4,
            "qvalue-lr": 3e-4,
            "entropy-scale-lr": 3e-4,
        }
    }.get(args_type)

  @staticmethod
  def make_env_kwargs(env_id):
    _ = env_id
    return dict(normalize_obs=False, normalize_ret=False)

  def make_runner(self, env, nlogs=1e5, **kwargs):
    with self.override_context(**kwargs):
      model = (self.get_arg("model") if self.has_arg("model")
               else SACModel.make(env.observation_space, env.action_space))
      policy = SACPolicy(model)
      runner_kwargs = self.get_arg_dict("num_train_steps",
                                        "storage_size", "storage_init_size",
                                        "batch_size", "steps_per_sample",
                                        "num_storage_samples")
      runner = make_sac_runner(env, policy, **runner_kwargs)
      return runner

  def make_trainer(self, runner, **kwargs):
    with self.override_context(**kwargs):
      model = runner.policy.model
      policy_opt = Adam(model.policy.parameters(), self.get_arg("policy_lr"))
      entropy_scale_opt = Adam((torch.zeros((), requires_grad=True),),
                               self.get_arg("entropy_scale_lr"))
      qvalue_opts = [Adam(qv.parameters(), self.get_arg("qvalue_lr"))
                     for qv in model.qvalues]
      trainer = SACTrainer(policy_opt, entropy_scale_opt, qvalue_opts)
      return trainer

  def make_alg(self, runner, trainer, **kwargs):
    with self.override_context(**kwargs):
      sac_kwargs = self.get_arg_dict(
          "target_update_coef", "target_update_period",
          "gamma", "reward_scale")
      log_entropy_scale = \
          trainer.entropy_scale_opt.param_groups[0]["params"][0]
      sac = SAC.make(runner, trainer, log_entropy_scale=log_entropy_scale,
                     **sac_kwargs)
      return sac
