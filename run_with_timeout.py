#!/usr/bin/env python3
"""Timeout wrapper for MPI commands."""

import subprocess
import signal
import sys

TIMEOUT_SECONDS = 60


def timeout_handler(signum, frame):
    """Handle timeout."""
    print(f"\n❌ TIMEOUT after {TIMEOUT_SECONDS} seconds", flush=True)
    sys.exit(124)


def main():
    """Run command with timeout."""
    if len(sys.argv) < 2:
        print("Usage: python run_with_timeout.py <command> [args...]")
        sys.exit(1)

    # Parse timeout if provided as first arg
    timeout = TIMEOUT_SECONDS
    if sys.argv[1].startswith("--timeout="):
        timeout = int(sys.argv[1].split("=")[1])
        cmd = sys.argv[2:]
    else:
        cmd = sys.argv[1:]

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        # Join command and run
        full_cmd = " ".join(cmd)
        result = subprocess.run(full_cmd, shell=True)
        signal.alarm(0)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        signal.alarm(0)
        sys.exit(130)


if __name__ == "__main__":
    main()

