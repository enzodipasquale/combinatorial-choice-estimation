#!/usr/bin/env python3
"""
Timeout wrapper for MPI runs to prevent hanging during debugging.
"""
import subprocess
import sys
import argparse


def main():
    parser = argparse.ArgumentParser(description='Run MPI command with timeout')
    parser.add_argument('--timeout', type=int, default=300,
                       help='Timeout in seconds (default: 300 = 5 minutes)')
    parser.add_argument('--mpi', type=int, default=10,
                       help='Number of MPI processes')
    parser.add_argument('script', type=str,
                       help='Python script to run')
    parser.add_argument('args', nargs=argparse.REMAINDER,
                       help='Arguments to pass to the script')
    
    args = parser.parse_args()
    
    cmd = ['mpirun', '-n', str(args.mpi), 'python', args.script] + args.args
    
    print(f"Running with timeout: {args.timeout}s")
    print(f"Command: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, timeout=args.timeout, check=False)
        sys.exit(result.returncode)
    except subprocess.TimeoutExpired:
        print(f"\nERROR: Command timed out after {args.timeout} seconds")
        print("This usually indicates a deadlock or infinite loop in MPI code.")
        sys.exit(124)  # Standard timeout exit code
    except KeyboardInterrupt:
        print("\nInterrupted by user")
        sys.exit(130)


if __name__ == '__main__':
    main()


