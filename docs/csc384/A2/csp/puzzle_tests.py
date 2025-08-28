# test_puzzle_csp_unit.py
import unittest
from puzzle_csp import binary_ne_grid, nary_ad_grid, caged_csp

class TestPuzzleCSP(unittest.TestCase):
    def setUp(self):
        self.board3 = [
            [3],
            [11, 12, 13, 6, 0],
            [21, 22, 23, 6, 0],
            [31, 32, 33, 6, 0]
        ]

    def test_binary_ne_grid(self):
        csp, var_array = binary_ne_grid(self.board3)
        self.assertEqual(len(var_array), 3)
        self.assertEqual(len(var_array[0]), 3)
        self.assertEqual(len(csp.get_all_vars()), 9)

    def test_nary_ad_grid(self):
        csp, var_array = nary_ad_grid(self.board3)
        self.assertEqual(len(var_array), 3)
        self.assertEqual(len(var_array[0]), 3)
        self.assertEqual(len(csp.get_all_vars()), 9)

    def test_caged_csp(self):
        csp, var_array = caged_csp(self.board3)
        self.assertEqual(len(var_array), 3)
        self.assertEqual(len(var_array[0]), 3)
        self.assertEqual(len(csp.get_all_vars()), 9)

if __name__ == '__main__':
    unittest.main()
