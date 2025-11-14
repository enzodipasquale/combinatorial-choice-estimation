#!/bin/bash
# Timeout wrapper for MPI commands

TIMEOUT_SECONDS=${1:-5}
shift

# Use gtimeout if available (from GNU coreutils via Homebrew), otherwise use Python timeout
if command -v gtimeout &> /dev/null; then
    gtimeout ${TIMEOUT_SECONDS} "$@"
else
    # Fallback: use Python to wrap the command
    python3 -c "
import subprocess
import signal
import sys
import os

def timeout_handler(signum, frame):
    print(f'\n❌ TIMEOUT after ${TIMEOUT_SECONDS} seconds', flush=True)
    sys.exit(124)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(${TIMEOUT_SECONDS})

try:
    result = subprocess.run('$@', shell=True)
    signal.alarm(0)
    sys.exit(result.returncode)
except KeyboardInterrupt:
    signal.alarm(0)
    sys.exit(130)
"
fi

