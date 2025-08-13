#!/bin/bash 

#SBATCH --job-name=greedy_experiment
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=00:05:00

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

srun ./run-gurobi.bash python experiment_inversion_greedy.py
