#!/usr/bin/env python3
"""
Test runner script for the polymarket-crypto-expiry-prediction project.

This script runs all tests and provides a convenient way to execute the test suite.
"""

import sys
import unittest
from pathlib import Path


def run_tests():
    """Run all tests in the tests directory."""
    # Add the project root to the Python path
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    # Discover and run tests
    loader = unittest.TestLoader()
    start_dir = project_root / "tests"
    suite = loader.discover(str(start_dir), pattern="test_*.py")

    # Run the tests
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)

    # Return exit code based on test results
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    exit_code = run_tests()
    sys.exit(exit_code)
