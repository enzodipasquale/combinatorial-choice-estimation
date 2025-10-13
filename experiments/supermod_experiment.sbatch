#!/bin/bash 

#SBATCH --job-name=supermod_experiment
#SBATCH --nodes=15
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=00:40:00

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

srun ./run-gurobi.bash python experiment_inversion_supermod.py

