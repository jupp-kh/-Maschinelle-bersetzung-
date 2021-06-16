"""
Automated test file for decoder.py
"""

import unittest
import sys

sys.path.insert(0, "..")
# import src.decoder as decoder
from pylint.lint import Run


class TestDecoder(unittest.TestCase):
    def test_code_integrity(self):
        """uses pylint to view code's quality"""
        Run(["--errors-only", "./../src/decoder.py"])


if __name__ == "__main__":
    unittest.main()
