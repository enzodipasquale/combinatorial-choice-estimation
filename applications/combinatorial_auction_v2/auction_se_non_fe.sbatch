#!/bin/bash 

#SBATCH --job-name=auction_se_non_fe
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=00:10:00
#SBATCH --output=logs/se-%j.out

# Usage: sbatch auction_se_non_fe.sbatch <delta> [winners] [hqdist]
#   delta: 2 or 4 (REQUIRED)
#   winners: optional, pass "winners" for winners-only sample
#   hqdist: optional, pass "hqdist" for HQ distance features
DELTA=${1:?Error: delta argument required. Usage: sbatch auction_se_non_fe.sbatch 4 [winners] [hqdist]}
WINNERS_FLAG=""
HQDIST_FLAG=""
for arg in "$@"; do
    if [ "$arg" == "winners" ]; then
        WINNERS_FLAG="--winners-only"
    fi
    if [ "$arg" == "hqdist" ]; then
        HQDIST_FLAG="--hq-distance"
    fi
done

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

# Step 1: Regenerate input_data with correct parameters
echo "Preparing input data with delta=${DELTA} ${WINNERS_FLAG} ${HQDIST_FLAG}..."
./run-gurobi.bash python prepare_data.py --delta ${DELTA} ${WINNERS_FLAG} ${HQDIST_FLAG}

# Step 2: Compute SE (MPI)
echo "Computing SE for delta=${DELTA} ${WINNERS_FLAG} ${HQDIST_FLAG}..."
srun ./run-gurobi.bash python compute_se_non_fe.py --delta ${DELTA} ${WINNERS_FLAG} ${HQDIST_FLAG}
