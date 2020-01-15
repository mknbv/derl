DERL is a a deep reinforcement learning package with a focus on
simplicity. It is built on top of PyTorch.

<p align="middle">
  <img src="./assets/beam-rider.gif" width=175 hspace=10/>
  <img src="./assets/breakout.gif" width=175 hspace=10/>
  <img src="./assets/qbert.gif" width=175 hspace=10/>
  <br/>
  <img src="./assets/half-cheetah.gif" width=180 hspace=10/>
  <img src="./assets/walker2d.gif" width=180 hsapce=10/>
  <p align="center">
    <sub><i>Parts of episodes where policies were learned with PPO.</i></sub>
  </p>
</p>

Currently implemented algorithms:

- A2C
- PPO
- DQN (n-step, double, dueling, with prioritized experience replay,
with noisy networks for exploration)

## Installation

The installation script will install torch but it might be better it
install it manually beforehand in order to insure proper system and
CUDA dependencies.

```{bash}
git clone https://github.com/MichaelKonobeev/derl.git
pip install -e derl
```

`gym[atari]` will be installed by `setup.py`, but you will need
to install other environment requirements (e.g. to use mujoco or pybullet)
separately.

Now you can run training:

```{bash}
derl-ppo --env-id BreakoutNoFrameskip-v4 --logdir logdir/breakout.00
```

Or if `gym[mujoco]` is installed:

```{bash}
derl-ppo --env-id HalfCheetah-v3 --logdir logdir/half-cheetah.00
```

The script automatically selects different hyperparameters for
atari and mujoco envs.
