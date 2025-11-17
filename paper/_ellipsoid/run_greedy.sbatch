#!/bin/bash 

#SBATCH --job-name=greedy_paper
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=38
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --time=3-00:00:00
#SBATCH --output=00_LOGS_SLURM/greedy_%j.out
#SBATCH --error=00_LOGS_SLURM/greedy_%j.err

module purge

cd /scratch/ed2189/combinatorial-choice-estimation/experiments_paper

# Use srun to launch the container wrapper, which will run the Python script inside Singularity
# This pattern matches the working auction example
# Large-scale run: 9 sizes × 2 replications × 3 methods = 54 runs
srun ../experiments/run-gurobi.bash python run_experiment.py greedy --mpi 150

