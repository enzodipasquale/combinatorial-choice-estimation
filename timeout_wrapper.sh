#!/bin/bash
# Improved timeout wrapper for MPI processes
timeout_seconds=$1
shift

# Start the command in background
"$@" &
pid=$!

# Function to kill process and all its children
kill_tree() {
    local parent_pid=$1
    # Kill all child processes
    pkill -P $parent_pid 2>/dev/null || true
    # Kill the parent process
    kill -9 $parent_pid 2>/dev/null || true
}

# Wait for timeout or completion
sleep $timeout_seconds
if kill -0 $pid 2>/dev/null; then
    echo "Timeout after ${timeout_seconds}s, killing process $pid and children"
    kill_tree $pid
    exit 124
else
    # Wait for the process to complete and get its exit code
    wait $pid
    exit_code=$?
    
    # Give a moment for any remaining child processes to clean up
    sleep 1
    
    # Kill any remaining child processes
    pkill -P $pid 2>/dev/null || true
    
    exit $exit_code
fi 