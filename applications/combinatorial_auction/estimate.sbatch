#!/bin/bash 

#SBATCH --job-name=point_est
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=01:00:00
#SBATCH --output=logs/point_est-%j.out

# Usage: sbatch estimate.sbatch <delta> [winners] [hqdist]
#   delta: 2 or 4 (default: 4)
#   winners: optional, pass "winners" for winners-only sample
#   hqdist: optional, pass "hqdist" for HQ distance features
DELTA=${1:-4}
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

# Step 1: Prepare data (no MPI needed, runs on single node)
echo "Preparing data with delta=${DELTA} ${WINNERS_FLAG} ${HQDIST_FLAG}..."
./run-gurobi.bash python prepare_data.py --delta ${DELTA} ${WINNERS_FLAG} ${HQDIST_FLAG}

# Step 2: Run estimation (MPI)
echo "Running estimation with delta=${DELTA} ${WINNERS_FLAG} ${HQDIST_FLAG}..."
srun ./run-gurobi.bash python run_estimation.py --delta ${DELTA} ${WINNERS_FLAG} ${HQDIST_FLAG}
