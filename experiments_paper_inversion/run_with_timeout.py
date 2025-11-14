#!/usr/bin/env python3
"""
Timeout wrapper for MPI runs to prevent hanging during debugging.
"""
import subprocess
import sys
import argparse
import os


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
    
    in_slurm = 'SLURM_JOB_ID' in os.environ
    # Check if we were launched via srun (SLURM_PROCID will be set)
    launched_via_srun = 'SLURM_PROCID' in os.environ

    if launched_via_srun:
        # Launched via srun - MPI processes are already set up by SLURM
        # Just run the Python script directly, mpi4py will use the existing MPI processes
        cmd = [sys.executable, args.script] + args.args
    elif in_slurm:
        # Under SLURM but not via srun - use mpirun
        # SLURM sets OMPI_COMM_WORLD_SIZE automatically, so we don't need to specify -n
        cmd = ['mpirun', sys.executable, args.script] + args.args
    else:
        # Local execution - specify number of processes
        cmd = ['mpirun', '-n', str(args.mpi), sys.executable, args.script] + args.args
    
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


