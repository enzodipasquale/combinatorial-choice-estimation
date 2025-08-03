#!/bin/bash

echo "Cleaning up orphaned MPI processes..."

# Kill any remaining Python processes from our experiments
echo "Killing Python experiment processes..."
pkill -f "row_generation_greedy_experiment.py" 2>/dev/null || echo "No row generation processes found"
pkill -f "greedy_ellipsoid_experiment.py" 2>/dev/null || echo "No ellipsoid processes found"

# Kill MPI launcher processes
echo "Killing MPI launcher processes..."
pkill -f "prterun" 2>/dev/null || echo "No prterun processes found"
pkill -f "mpirun" 2>/dev/null || echo "No mpirun processes found"

# Kill any remaining Python processes that might be MPI ranks
echo "Checking for remaining Python MPI processes..."
ps aux | grep python | grep -E "(experiment|mpi)" | grep -v grep

echo "Cleanup complete!"
echo ""
echo "To check if cleanup was successful, run:"
echo "ps aux | grep python | grep -v grep" 