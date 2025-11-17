#!/bin/bash 

#SBATCH --job-name=supermod_paper
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=33
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --time=00:10:00
#SBATCH --output=__logs_slurm/supermod_%j.out
#SBATCH --error=__logs_slurm/supermod_%j.err

module purge

cd /scratch/ed2189/combinatorial-choice-estimation/experiments_paper

srun ../experiments/run-gurobi.bash python run_experiment.py supermod  

