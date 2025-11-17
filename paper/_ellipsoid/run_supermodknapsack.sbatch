#!/bin/bash 

#SBATCH --job-name=supermodknapsack_paper
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=38
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --time=5-00:00:00
#SBATCH --output=00_LOGS_SLURM/supermodknapsack_%j.out
#SBATCH --error=00_LOGS_SLURM/supermodknapsack_%j.err

module purge

cd /scratch/ed2189/combinatorial-choice-estimation/experiments_paper

# run_experiment.py orchestrates the pipeline and calls mpirun internally
# mpirun will detect SLURM allocation and use the allocated tasks
# Large-scale run: 9 sizes × 50 replications × 3 methods = 1350 runs
# Conservative timeout: 60 minutes per size (supermod variants can be very slow for large sizes)
srun ../experiments/run-gurobi.bash python run_experiment.py supermodknapsack --mpi 150

