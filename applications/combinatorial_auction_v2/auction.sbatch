#!/bin/bash 

#SBATCH --job-name=auction_estimation
#SBATCH --nodes=35
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --output=logs/estimation-%j.out

# Usage: sbatch auction.sbatch <delta> [winners]
#   delta: 2 or 4 (default: 4)
#   winners: optional, pass "winners" for winners-only sample
DELTA=${1:-4}
WINNERS_FLAG=""
if [ "$2" == "winners" ]; then
    WINNERS_FLAG="--winners-only"
fi

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

# Step 1: Prepare data (no MPI needed, runs on single node)
echo "Preparing data with delta=${DELTA} ${WINNERS_FLAG}..."
./run-gurobi.bash python prepare_data.py --delta ${DELTA} ${WINNERS_FLAG}

# Step 2: Run estimation (MPI)
echo "Running estimation with delta=${DELTA} ${WINNERS_FLAG}..."
srun ./run-gurobi.bash python run_estimation.py --delta ${DELTA} ${WINNERS_FLAG}









