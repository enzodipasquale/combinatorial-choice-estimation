#!/bin/bash 

#SBATCH --job-name=se_bootstrap
#SBATCH --nodes=15
#SBATCH --ntasks-per-node=17
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=02:00:00
#SBATCH --output=../logs/se_bootstrap-%j.out

# All parameters are read from config.yaml
# Edit config.yaml to change delta, num_samples, warmstart, etc.

module purge
export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

# Prepare data (reads parameters from config)
echo "Preparing data..."
../run-gurobi.bash python ../data/prepare_data.py --config config.yaml

# Run bootstrap estimation (independent pipeline)
echo "Running bootstrap SE estimation..."
srun ../run-gurobi.bash python compute_se_bootstrap.py
