#!/bin/bash 

#SBATCH --job-name=auction_bayes_boot
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=logs/bayes_boot-%j.out

# Compute SE using Bayesian Bootstrap with model warm-start (~20x faster)
#
# Usage: sbatch auction_se_bayesian_boot.sbatch <delta> [num_bootstrap] [winners] [hqdist]
#   delta: 2 or 4 (REQUIRED)
#   num_bootstrap: number of bootstrap samples (default: 200)
#   winners: optional, pass "winners" for winners-only sample
#   hqdist: optional, pass "hqdist" for HQ distance features

DELTA=${1:?Error: delta argument required. Usage: sbatch auction_se_bayesian_boot.sbatch 4 [200] [winners] [hqdist]}
NUM_BOOTSTRAP=${2:-200}
WINNERS_FLAG=""
HQDIST_FLAG=""

for arg in "${@:3}"; do
    if [ "$arg" == "winners" ]; then
        WINNERS_FLAG="--winners-only"
    fi
    if [ "$arg" == "hqdist" ]; then
        HQDIST_FLAG="--hq-distance"
    fi
done

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

# Ensure input data exists
echo "Checking input data with delta=${DELTA} ${WINNERS_FLAG} ${HQDIST_FLAG}..."
./run-gurobi.bash python prepare_data.py --delta ${DELTA} ${WINNERS_FLAG} ${HQDIST_FLAG}

# Compute SE via Bayesian Bootstrap with model_strip warm-start
echo ""
echo "Computing SE via Bayesian Bootstrap..."
echo "  delta=${DELTA}, num_bootstrap=${NUM_BOOTSTRAP}, warmstart=model_strip"
echo ""
srun ./run-gurobi.bash python compute_se_bayesian_boot.py \
    --delta ${DELTA} \
    --num-bootstrap ${NUM_BOOTSTRAP} \
    --warmstart model_strip \
    ${WINNERS_FLAG} \
    ${HQDIST_FLAG}
