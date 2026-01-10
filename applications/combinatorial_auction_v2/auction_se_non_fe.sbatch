#!/bin/bash 

#SBATCH --job-name=auction_se_non_fe
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=00:10:00
#SBATCH --output=logs/se-%j.out

# Distance parameter delta (2 or 4) - REQUIRED
DELTA=${1:?Error: delta argument required. Usage: sbatch auction_se_non_fe.sbatch 4}

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

# Step 1: Regenerate input_data with correct delta (ensures features match)
echo "Preparing input data with delta=${DELTA}..."
./run-gurobi.bash python prepare_data.py --delta ${DELTA}

# Step 2: Compute SE (MPI)
echo "Computing SE for delta=${DELTA}..."
srun ./run-gurobi.bash python compute_se_non_fe.py --delta ${DELTA}
