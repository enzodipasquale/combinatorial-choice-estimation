#!/bin/bash 

#SBATCH --job-name=se_sandwich
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=../logs/se_sandwich-%j.out

# All parameters are read from config.yaml
# Edit config.yaml to change delta, n_simulations, etc.

module purge
export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

# Prepare data
echo "Preparing data..."
../run-gurobi.bash python ../data/prepare_data.py --config config.yaml

# Compute sandwich SE
echo "Computing sandwich SE..."
srun ../run-gurobi.bash python compute_se_sandwich.py
