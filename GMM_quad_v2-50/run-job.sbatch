#!/bin/bash 

#SBATCH --job-name=GMM_quad-v2_50
#SBATCH --nodes=15
#SBATCH --ntasks-per-node=20 
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=06:00:00

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

srun ./run-gurobi.bash python job-main.py




