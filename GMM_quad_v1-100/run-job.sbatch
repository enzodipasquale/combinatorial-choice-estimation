#!/bin/bash 

#SBATCH --job-name=GMM_quad-v1_100
#SBATCH --nodes=25
#SBATCH --ntasks-per-node=25 
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --time=06:00:00

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

srun ./run-gurobi.bash python job-main.py




