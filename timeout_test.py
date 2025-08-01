#!/usr/bin/env python3
import subprocess
import sys
import time

def run_with_timeout(cmd, timeout=5):
    print(f"Running: {' '.join(cmd)}")
    print(f"Timeout: {timeout} seconds")
    
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    try:
        stdout, stderr = proc.communicate(timeout=timeout)
        print(f"SUCCESS: Exit code {proc.returncode}")
        print("STDOUT:")
        print(stdout.decode())
        print("STDERR:")
        print(stderr.decode())
        return proc.returncode == 0
    except subprocess.TimeoutExpired:
        print("TIMEOUT - killing process")
        proc.kill()
        stdout, stderr = proc.communicate()
        print("STDOUT:")
        print(stdout.decode())
        print("STDERR:")
        print(stderr.decode())
        return False

if __name__ == "__main__":
    cmd = ['mpirun', '-n', '10', 'python3', '-m', 'pytest', 'bundlechoice/tests/test_data_manager_mpi.py', '-v', '-s']
    success = run_with_timeout(cmd, timeout=3)
    sys.exit(0 if success else 1) 