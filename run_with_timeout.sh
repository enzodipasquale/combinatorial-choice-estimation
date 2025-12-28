#!/bin/bash
# Timeout wrapper for MPI commands

TIMEOUT_SECONDS=${1:-60}
shift

# Use timeout if available (GNU coreutils), otherwise use Python fallback
if command -v timeout &> /dev/null; then
    timeout ${TIMEOUT_SECONDS} "$@"
elif command -v gtimeout &> /dev/null; then
    gtimeout ${TIMEOUT_SECONDS} "$@"
else
    # Fallback: use Python to wrap the command
    python3 -c "
import subprocess
import signal
import sys

timeout_sec = ${TIMEOUT_SECONDS}
args = sys.argv[1:]

def timeout_handler(signum, frame):
    print(f'\n❌ TIMEOUT after {timeout_sec} seconds', flush=True)
    sys.exit(124)

signal.signal(signal.SIGALRM, timeout_handler)
signal.alarm(timeout_sec)

try:
    result = subprocess.run(args)
    signal.alarm(0)
    sys.exit(result.returncode)
except KeyboardInterrupt:
    signal.alarm(0)
    sys.exit(130)
" "$@"
fi

