#!/usr/bin/env python3
"""
Automated Test Runner for Stock Prediction Platform.
Runs all unit and integration test suites.
"""

import sys
import unittest

def main():
    print("=" * 70)
    print("   STOCK PREDICTION PROGRAM - AUTOMATED TEST SUITE")
    print("=" * 70)
    
    loader = unittest.TestLoader()
    suite = loader.discover("tests", pattern="test_*.py")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    print("\n" + "=" * 70)
    if result.wasSuccessful():
        print(f"✅ ALL {result.testsRun} TESTS PASSED SUCCESSFULLY.")
        print("=" * 70)
        sys.exit(0)
    else:
        print(f"❌ TEST SUITE FAILED: {len(result.failures)} failures, {len(result.errors)} errors.")
        print("=" * 70)
        sys.exit(1)

if __name__ == "__main__":
    main()
