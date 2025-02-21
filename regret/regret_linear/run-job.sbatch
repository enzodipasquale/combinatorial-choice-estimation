#!/bin/bash 

#SBATCH --job-name=regret_linear
#SBATCH --nodes=5 # 5 nodes
#SBATCH --ntasks-per-node=10 # 10 tasks per node
#SBATCH --cpus-per-task=1
#SBATCH --mem=5G
#SBATCH --time=07:00:00

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

srun ./run-gurobi.bash python job-main.py




