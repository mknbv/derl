from distutils.core import setup

setup(
    name="derl",
    version="0.1dev",
    packages=["derl"],
    install_requires=[
        "numpy",
        "opencv-python",
        "gym[atari]>=0.11",
        "tqdm",
    ],
    license="MIT",
    long_description="DERL is a Deep Reinforcement Learning package"
)
