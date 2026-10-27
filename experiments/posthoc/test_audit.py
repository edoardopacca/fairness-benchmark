import unittest
import numpy as np
from .audit import permutation_p


class PermutationAuditTests(unittest.TestCase):
    def test_toy_enumeration(self):
        # Four observations, two groups of size two, two positives:
        # two of six equally likely arrangements have all positives in one group.
        p = permutation_p([2, 2], [2, 0], permutations=99999)
        self.assertLess(abs(p - 1/3), .008)
        self.assertEqual(permutation_p([2, 2], [1, 1]), 1)

    def test_constant_predictions(self):
        self.assertEqual(permutation_p([10, 20, 30], [0, 0, 0]), 1)
        self.assertEqual(permutation_p([10, 20, 30], [10, 20, 30]), 1)

    def test_reproducibility_and_group_symmetry(self):
        a = permutation_p([100, 200], [10, 180])
        b = permutation_p([100, 200], [10, 180])
        self.assertEqual(a, b)
        self.assertLess(a, .001)
        self.assertLess(permutation_p([200, 100], [180, 10]), .001)


if __name__ == "__main__":
    unittest.main()
