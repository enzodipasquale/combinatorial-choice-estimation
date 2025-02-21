#!/bin/bash 

#SBATCH --job-name=GMM_quad-v3_200
#SBATCH --nodes=25
#SBATCH --ntasks-per-node=25 
#SBATCH --cpus-per-task=1
#SBATCH --mem=25G
#SBATCH --time=24:00:00

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

srun ./run-gurobi.bash python job-main.py




