import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from .util import unittest

if __name__ == '__main__':
    suite = unittest.TestLoader().discover("tests")
    ret = unittest.TextTestRunner(verbosity=2).run(suite)
    if ret.wasSuccessful():
        sys.exit(0)
    sys.exit(1)
