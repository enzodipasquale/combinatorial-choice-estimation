#!/bin/bash 

#SBATCH --job-name=regret_quad-2
#SBATCH --nodes=6 # 5 nodes
#SBATCH --ntasks-per-node=22 # 10 tasks per node
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=12:00:00

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

srun ./run-gurobi.bash python job-main.py




