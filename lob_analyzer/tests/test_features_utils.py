import unittest
from lob_analyzer.utils.features_utils import safe_float, safe_int

class TestFeaturesUtils(unittest.TestCase):

    def test_safe_float(self):
        self.assertEqual(safe_float("123.45"), 123.45)
        self.assertEqual(safe_float("invalid", default=0.0), 0.0)
        self.assertEqual(safe_float(None, default=0.0), 0.0)

    def test_safe_int(self):
        self.assertEqual(safe_int("123"), 123)
        self.assertEqual(safe_int("invalid", default=0), 0)
        self.assertEqual(safe_int(None, default=0), 0)

if __name__ == "__main__":
    unittest.main()