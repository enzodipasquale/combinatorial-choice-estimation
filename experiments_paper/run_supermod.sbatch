#!/bin/bash 

#SBATCH --job-name=supermod_paper
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=00:10:00
#SBATCH --output=supermod_%j.out
#SBATCH --error=supermod_%j.err

module purge

cd /scratch/ed2189/combinatorial-choice-estimation/experiments_paper

# run_experiment.py orchestrates the pipeline and calls mpirun internally
# mpirun will detect SLURM allocation and use the allocated tasks
# 4 minute timeout per size to prevent hanging
../experiments/run-gurobi.bash python run_experiment.py supermod --mpi 10 --timeout 240

