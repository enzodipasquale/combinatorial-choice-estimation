#!/bin/bash 

#SBATCH --job-name=GMM_quad-v1_100
#SBATCH --nodes=35  
#SBATCH --ntasks-per-node=30 
#SBATCH --cpus-per-task=1
#SBATCH --mem=30G
#SBATCH --time=00:30:00

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

srun ./run-gurobi.bash python job-main.py




