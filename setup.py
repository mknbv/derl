from distutils.core import setup

setup(
    name="derl",
    version="0.1dev",
    description="Deep Reinforcement Learning package",
    author="Mikhail Konobeev",
    author_email="konobeev.michael@gmail.com",
    license="MIT",
    packages=["derl"],
    scripts=[
        "derl/scripts/derl-a2c",
    ],
    install_requires=[
        "gym[atari]>=0.11",
        "numpy",
        "opencv-python",
        "tensorflow-probability",
        "tqdm",
    ],
    long_description="DERL is a Deep Reinforcement Learning package"
)
