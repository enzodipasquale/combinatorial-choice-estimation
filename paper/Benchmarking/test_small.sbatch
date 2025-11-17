#!/bin/bash 

#SBATCH --job-name=test_small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:05:00
#SBATCH --output=__logs_slurm/test_small_%j.out
#SBATCH --error=__logs_slurm/test_small_%j.err

module purge

cd /scratch/ed2189/combinatorial-choice-estimation/experiments_paper

srun ../experiments/run-gurobi.bash python run_experiment.py greedy --mpi 5 --sizes test_small_sizes.yaml --config experiments/greedy/config.yaml

