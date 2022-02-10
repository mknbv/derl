# pylint: disable=missing-docstring
from unittest import TestCase
import numpy as np
import numpy.testing as nt
from derl.runners.storage import InteractionStorage


class InteractionStorageTest(TestCase):
  four_interactions_batch = dict(observations=np.asarray([0, 1, 2, 3]),
                                 actions=np.asarray([-1, -2, -3, -4]),
                                 rewards=np.asarray([0.1, 0.2, 0.3, 0.4]),
                                 resets=np.asarray([False, True, False, True]))

  def test_sample_not_full(self):
    storage = InteractionStorage(capacity=10, nstep=1)
    interactions = self.four_interactions_batch
    storage.add_batch(**interactions)
    self.assertEqual(storage.index, 4)
    nt.assert_equal(storage.arrays.observations[:storage.index],
                    interactions["observations"])
    nt.assert_equal(storage.arrays.actions[:storage.index],
                    interactions["actions"])
    nt.assert_allclose(storage.arrays.rewards[:storage.index],
                       interactions["rewards"])
    nt.assert_equal(storage.arrays.resets[:storage.index],
                    interactions["resets"])
    np.random.seed(3)
    nt.assert_equal(np.random.randint(3), 2)
    np.random.seed(3)
    sample = storage.sample(1)
    nt.assert_equal(sample["observations"], [2])
    nt.assert_equal(sample["actions"], [-3])
    nt.assert_allclose(sample["rewards"], [[0.3]])
    nt.assert_equal(sample["resets"], [[False]])
    nt.assert_equal(sample["next_observations"], [3])

  def test_sample_full(self):
    storage = InteractionStorage(capacity=4, nstep=2)
    interactions = self.four_interactions_batch
    storage.add_batch(**interactions)
    self.assertEqual(storage.index, 0)
    nt.assert_equal(storage.arrays.observations, interactions["observations"])
    nt.assert_equal(storage.arrays.actions, interactions["actions"])
    nt.assert_allclose(storage.arrays.rewards, interactions["rewards"])
    nt.assert_equal(storage.arrays.resets, interactions["resets"])
    np.random.seed(0)
    nt.assert_equal(np.random.randint(2, size=2), [0, 1])
    np.random.seed(0)
    sample = storage.sample(size=2)
    nt.assert_equal(sample["observations"], [0, 1])
    nt.assert_equal(sample["actions"], [-1, -2])
    nt.assert_allclose(sample["rewards"], [[0.1, 0.2], [0.2, 0.3]])
    nt.assert_equal(sample["resets"], [[False, True], [True, False]])
    nt.assert_equal(sample["next_observations"], [2, 3])

  def test_sample_cycle(self):
    storage = InteractionStorage(capacity=3, nstep=2)
    storage.add_batch(**self.four_interactions_batch)
    nt.assert_equal(storage.arrays.observations, [3, 1, 2])
    nt.assert_equal(storage.arrays.actions, [-4, -2, -3])
    nt.assert_allclose(storage.arrays.rewards, [0.4, 0.2, 0.3])
    nt.assert_equal(storage.arrays.resets, [True, True, False])
    np.random.seed(0)
    nt.assert_equal(np.random.randint(storage.capacity - storage.nstep), 0)
    np.random.seed(0)
    sample = storage.sample(1)
    nt.assert_equal(sample["observations"], [1])
    nt.assert_equal(sample["actions"], [-2])
    nt.assert_allclose(sample["rewards"], [[0.2, 0.3]])
    nt.assert_equal(sample["resets"], [[True, False]])
    nt.assert_equal(sample["next_observations"], [3])
