from distutils.core import setup

setup(
    name="derl",
    version="0.1dev",
    description="Deep Reinforcement Learning package",
    author="Mikhail Konobeev",
    author_email="konobeev.michael@gmail.com",
    url="https://github.com/MichaelKonobeev/derl/",
    license="MIT",
    packages=["derl"],
    scripts=["derl/scripts/derl"],
    install_requires=[
        "gym[atari]>=0.11,<0.15",
        "pybullet",
        "scipy==1.5.0",
        "numpy==1.16.4",
        "opencv-python",
        "tensorboard==1.14",
        "torch>=1.4.0,<1.6.0",
        "tqdm",
    ],
    long_description="DERL is a Deep Reinforcement Learning package"
)
