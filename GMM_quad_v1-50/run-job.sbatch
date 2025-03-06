#!/bin/bash 

#SBATCH --job-name=GMM_quad-v1_50
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=10 
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=00:10:00

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

srun ./run-gurobi.bash python job-main.py




