"""
Automated test file for metrics.py
"""

import unittest
import os
import sys
import inspect

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


if __name__ == "__main__":
    unittest.main()
