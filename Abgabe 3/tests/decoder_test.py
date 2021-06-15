"""
Automated test file for decoder.py
"""

import unittest
import decoder
from pylint.lint import Run


class TestDecoder(unittest.TestCase):
    def test_code_integrity(self):
        """uses pylint to view code's quality"""
        Run(["--errors-only", "decoder.py"])


if __name__ == "__main__":
    unittest.main()
