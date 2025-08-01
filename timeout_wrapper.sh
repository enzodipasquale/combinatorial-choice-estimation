#!/bin/bash
# Simple timeout wrapper
timeout_seconds=$1
shift

# Start the command in background
"$@" &
pid=$!

# Wait for timeout or completion
sleep $timeout_seconds
if kill -0 $pid 2>/dev/null; then
    echo "Timeout after ${timeout_seconds}s, killing process $pid"
    kill -9 $pid 2>/dev/null
    exit 124
else
    wait $pid
    exit $?
fi 