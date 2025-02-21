#!/bin/bash 

#SBATCH --job-name=GMM_linear
#SBATCH --nodes=10 # 5 nodes
#SBATCH --ntasks-per-node=20 # 10 tasks per node
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=12:00:00

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

srun ./run-gurobi.bash python job-main.py




