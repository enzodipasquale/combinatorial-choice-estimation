#!/bin/bash 

#SBATCH --job-name=point_est
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=00:05:00
#SBATCH --output=../logs/point_est-%j.out
#SBATCH --account=torch_pr_531_general

# All parameters are read from config.yaml
# Edit config.yaml to change delta, winners_only, hq_distance, etc.

module purge
export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

# Prepare data (reads parameters from config)
echo "Preparing data..."
../run-gurobi.bash python ../data/prepare_data.py --config config.yaml

# Run estimation
echo "Running estimation..."
srun ../run-gurobi.bash python run_estimation.py
