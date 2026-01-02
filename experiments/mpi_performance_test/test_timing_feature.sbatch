#!/bin/bash

#SBATCH --job-name=timing_test_feature
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=00:03:00
#SBATCH --output=slurm-timing-test-feature-%j.out
#SBATCH --error=slurm-timing-test-feature-%j.err

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

PROJECT_ROOT="/scratch/ed2189/combinatorial-choice-estimation"

cd "$PROJECT_ROOT/applications/combinatorial_auction_v2"

srun ./run-gurobi.bash python "$PROJECT_ROOT/experiments/mpi_performance_test/run_timing_test.py"

