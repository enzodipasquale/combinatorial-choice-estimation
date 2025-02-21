#!/bin/bash 

#SBATCH --job-name=GMM_quad-v1
#SBATCH --nodes=12 # 
#SBATCH --ntasks-per-node=24 
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=06:00:00

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

srun ./run-gurobi.bash python job-main.py




