#!/bin/bash 

#SBATCH --job-name=supermodknapsack_paper
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=00:10:00
#SBATCH --output=supermodknapsack_%j.out
#SBATCH --error=supermodknapsack_%j.err

module purge

cd /scratch/ed2189/combinatorial-choice-estimation/experiments_paper

# run_experiment.py orchestrates the pipeline and calls mpirun internally
# mpirun will detect SLURM allocation and use the allocated tasks
# 4 minute timeout per size to prevent hanging
../experiments/run-gurobi.bash python run_experiment.py supermodknapsack --mpi 10 --timeout 240

