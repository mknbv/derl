""" derl.alg subpackage initialization. """
from .common import Loss, Trainer, Alg, r_squared
from .a2c import A2CLoss, A2C
from .dqn import TargetUpdater, DQNLoss, DQN
from .ppo import PPOLoss, PPO
from .sac import SACLoss
