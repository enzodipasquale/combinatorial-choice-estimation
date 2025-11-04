#!/usr/bin/env python3
"""
Run all tests with timeout wrapper.

Usage:
    mpirun -n 10 python run_tests_with_timeout.py [timeout_seconds]
"""

import sys
import subprocess
import signal
import os
from pathlib import Path


def run_with_timeout(cmd, timeout_seconds=300):
    """Run command with timeout."""
    def timeout_handler(signum, frame):
        raise TimeoutError(f"Command timed out after {timeout_seconds} seconds")
    
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout_seconds)
    
    try:
        result = subprocess.run(cmd, shell=True, check=False)
        signal.alarm(0)
        return result.returncode
    except TimeoutError as e:
        print(f"\n❌ {e}", file=sys.stderr)
        signal.alarm(0)
        return 124
    except KeyboardInterrupt:
        signal.alarm(0)
        return 130


def main():
    """Run all tests with timeout."""
    timeout = int(sys.argv[1]) if len(sys.argv) > 1 else 300  # Default 5 minutes per test
    
    test_dir = Path(__file__).parent / "bundlechoice" / "tests"
    test_files = sorted(test_dir.glob("test_*.py"))
    
    if not test_files:
        print("❌ No test files found!")
        sys.exit(1)
    
    print(f"🧪 Running {len(test_files)} test files with {timeout}s timeout each...")
    print(f"📁 Test directory: {test_dir}\n")
    
    failed_tests = []
    passed_tests = []
    timed_out_tests = []
    
    for i, test_file in enumerate(test_files, 1):
        test_name = test_file.stem
        print(f"[{i}/{len(test_files)}] Running {test_name}...", end=" ", flush=True)
        
        # Run pytest on this specific test file
        cmd = f"pytest {test_file} -v"
        exit_code = run_with_timeout(cmd, timeout_seconds=timeout)
        
        if exit_code == 124:
            print("⏱️  TIMEOUT")
            timed_out_tests.append(test_name)
        elif exit_code == 0:
            print("✅ PASSED")
            passed_tests.append(test_name)
        else:
            print("❌ FAILED")
            failed_tests.append(test_name)
    
    # Summary
    print("\n" + "="*60)
    print("📊 TEST SUMMARY")
    print("="*60)
    print(f"✅ Passed:  {len(passed_tests)}/{len(test_files)}")
    print(f"❌ Failed:  {len(failed_tests)}/{len(test_files)}")
    print(f"⏱️  Timeout: {len(timed_out_tests)}/{len(test_files)}")
    
    if passed_tests:
        print(f"\n✅ Passed tests: {', '.join(passed_tests)}")
    if failed_tests:
        print(f"\n❌ Failed tests: {', '.join(failed_tests)}")
    if timed_out_tests:
        print(f"\n⏱️  Timed out tests: {', '.join(timed_out_tests)}")
    
    if failed_tests or timed_out_tests:
        sys.exit(1)
    else:
        print("\n🎉 All tests passed!")
        sys.exit(0)


if __name__ == "__main__":
    main()

