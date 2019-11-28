import unittest
import numpy as np

from .game import StateBatch


class TestStateBatch(unittest.TestCase):
    def test_compute_batch(self):
        sb = StateBatch()

        how_many = 10
        hidden_states = [np.random.rand(100) for _ in range(how_many)]
        # trajectory =


if __name__ == '__main__':
    unittest.main()
