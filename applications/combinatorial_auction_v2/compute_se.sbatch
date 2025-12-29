#!/bin/bash 

#SBATCH --job-name=combinatorial_auction_v2_sandwich_se
#SBATCH --nodes=13
#SBATCH --ntasks-per-node=16
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=00:05:00
#SBATCH --output=slurm-sandwich-se-%j.out
#SBATCH --error=slurm-sandwich-se-%j.err

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

srun ./run-gurobi.bash python compute_sandwich_se_v2.py

