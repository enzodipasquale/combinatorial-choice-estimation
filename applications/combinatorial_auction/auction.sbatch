#!/bin/bash 

#SBATCH --job-name=combinatorial_auction
#SBATCH --nodes=15
#SBATCH --ntasks-per-node=15
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=15:00:00

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

srun ./run-gurobi.bash python run_estimation.py




