"""
Automated test file for decoder.py
"""

import os
import unittest


class TestDecoder(unittest.TestCase):
    """implements unittest.TestCase"""

    def test_code_integrity(self):
        """uses pylint to view code's quality"""
        os.system("pylint src/decoder.py")


if __name__ == "__main__":
    unittest.main()
