#!/bin/bash 

#SBATCH --job-name=test_greedy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=00:02:00
#SBATCH --output=test_greedy_%j.out
#SBATCH --error=test_greedy_%j.err

module purge

cd /scratch/ed2189/combinatorial-choice-estimation/experiments_paper

echo "Starting test run at $(date)"
echo "Working directory: $(pwd)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Test with minimal settings - create minimal test_sizes.yaml if needed
if [ ! -f greedy/test_sizes.yaml ]; then
  echo "Creating test_sizes.yaml..."
  cat > greedy/test_sizes.yaml << EOF
# Minimal test sizes for debugging
sizes_grid:
  agents: [5]
  items: [5]
EOF
fi

# Temporarily use test_sizes.yaml
cp greedy/sizes.yaml greedy/sizes.yaml.backup 2>/dev/null || true
cp greedy/test_sizes.yaml greedy/sizes.yaml

# Use 1 minute timeout (60 seconds) for quick debugging
TIMEOUT=60
../experiments/run-gurobi.bash python run_experiment.py greedy --mpi 4 --timeout $TIMEOUT 2>&1

EXIT_CODE=$?

# Restore original sizes.yaml
if [ -f greedy/sizes.yaml.backup ]; then
  mv greedy/sizes.yaml.backup greedy/sizes.yaml
else
  # Restore from git if backup doesn't exist
  git checkout greedy/sizes.yaml 2>/dev/null || echo "Warning: Could not restore sizes.yaml"
fi

if [ $EXIT_CODE -eq 124 ]; then
  echo "WARNING: Test timed out after ${TIMEOUT} seconds"
elif [ $EXIT_CODE -ne 0 ]; then
  echo "ERROR: Test failed with exit code: $EXIT_CODE"
else
  echo "Test completed successfully"
fi

echo "Test completed at $(date)"
exit $EXIT_CODE

