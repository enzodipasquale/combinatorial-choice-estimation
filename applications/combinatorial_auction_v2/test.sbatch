#!/bin/bash 

# SBATCH --job-name=combinatorial_auction_v2_test
# SBATCH --nodes=1
# SBATCH --ntasks-per-node=1
# SBATCH --cpus-per-task=1
# SBATCH --mem=5G
# SBATCH --time=00:01:00

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

srun ./run-gurobi.bash python run_estimation.py




