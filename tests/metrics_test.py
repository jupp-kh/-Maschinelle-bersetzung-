"""
Automated test file for metrics.py
"""

import unittest
import os
import sys
import inspect

# TODO improve this
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)


import src.metrics as metrics


class TestMetrics(unittest.TestCase):
    def test_met_bleu(self):
        src_str = ["the cat is on the mat"]
        tar_str = ["the the the the the the the"]

        # test accuracy of answer up to two decimal points
        self.assertEqual(
            round(metrics.met_bleu(src_str, tar_str, 4), 2),
            0.84,
            msg="Bleu is inaccurate",
        )

        # check behaviour if inputs for bleu are same
        for i in range(5):
            self.assertEqual(
                metrics.met_bleu(src_str, src_str, i),
                1,
                msg="Bleu fails when given two equal sentences at " + str(i),
            )


"""
assertEqual(a, b) 	a == b

assertNotEqual(a, b)	a != b

assertTrue(x)		bool(x) is True

assertFalse(x)		bool(x) is False

assertIs(a, b)		a is b

assertIsNot(a, b)	a is not b

assertIsNone(x)		x is None

assertIsNotNone(x)	x is not None

assertIn(a, b)		a in b

assertNotIn(a, b)	a not in b

assertIsInstance(a, b)

isinstance(a, b)

assertNotIsInstance(a, b) 	not isinstance(a, b)
"""


if __name__ == "__main__":
    unittest.main()
