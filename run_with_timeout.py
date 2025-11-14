#!/usr/bin/env python3
"""Timeout wrapper for MPI commands."""

import subprocess
import signal
import sys

TIMEOUT_SECONDS = 60


def main():
    """Run command with timeout."""
    if len(sys.argv) < 2:
        print("Usage: python run_with_timeout.py [--timeout=SECONDS] <command> [args...]")
        sys.exit(1)

    # Parse timeout
    timeout = TIMEOUT_SECONDS
    cmd_start = 1
    
    if sys.argv[1].startswith("--timeout="):
        timeout = int(sys.argv[1].split("=")[1])
        cmd_start = 2
    elif sys.argv[1] == "--timeout" and len(sys.argv) > 2:
        timeout = int(sys.argv[2])
        cmd_start = 3
    
    if cmd_start >= len(sys.argv):
        print("Error: No command provided", file=sys.stderr)
        sys.exit(1)
    
    cmd = sys.argv[cmd_start:]

    def timeout_handler(signum, frame):
        """Handle timeout."""
        print(f"\n❌ TIMEOUT after {timeout} seconds", flush=True)
        sys.exit(124)

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(timeout)

    try:
        full_cmd = " ".join(cmd)
        result = subprocess.run(full_cmd, shell=True)
        signal.alarm(0)
        sys.exit(result.returncode)
    except KeyboardInterrupt:
        signal.alarm(0)
        sys.exit(130)


if __name__ == "__main__":
    main()

