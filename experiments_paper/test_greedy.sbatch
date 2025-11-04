#!/bin/bash 

#SBATCH --job-name=test_greedy
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=00:05:00
#SBATCH --output=test_greedy_%j.out
#SBATCH --error=test_greedy_%j.err

module purge

cd /scratch/ed2189/combinatorial-choice-estimation/experiments_paper

echo "Starting test run at $(date)"
echo "Working directory: $(pwd)"
echo "SLURM_JOB_ID: $SLURM_JOB_ID"

# Test with minimal settings - override sizes.yaml with test_sizes.yaml
export NUM_AGENTS=5
export NUM_ITEMS=5

# Temporarily use test_sizes.yaml
cp greedy/sizes.yaml greedy/sizes.yaml.backup
cp greedy/test_sizes.yaml greedy/sizes.yaml

# Use 4 minute timeout (240 seconds) for each size
../experiments/run-gurobi.bash python run_experiment.py greedy --mpi 10 --timeout 240 2>&1

EXIT_CODE=$?

# Restore original sizes.yaml
mv greedy/sizes.yaml.backup greedy/sizes.yaml

if [ $EXIT_CODE -eq 124 ]; then
  echo "WARNING: Test timed out after ${TIMEOUT} seconds"
elif [ $EXIT_CODE -ne 0 ]; then
  echo "ERROR: Test failed with exit code: $EXIT_CODE"
else
  echo "Test completed successfully"
fi

echo "Test completed at $(date)"
exit $EXIT_CODE

