#!/bin/bash 

#SBATCH --job-name=auction_estimation
#SBATCH --nodes=35
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=05:00:00
#SBATCH --output=slurm-estimation-%j.out

# Distance parameter delta (2 or 4)
DELTA=${1:-4}

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

# Step 1: Prepare data (no MPI needed, runs on single node)
echo "Preparing data with delta=${DELTA}..."
./run-gurobi.bash python prepare_data.py --delta ${DELTA}

# Step 2: Run estimation (MPI)
echo "Running estimation with delta=${DELTA}..."
srun ./run-gurobi.bash python run_estimation.py --delta ${DELTA}









