#!/bin/bash 

#SBATCH --job-name=combinatorial_auction_v2_bootstrap
#SBATCH --nodes=15
#SBATCH --ntasks-per-node=15
#SBATCH --cpus-per-task=1
#SBATCH --mem=50G
#SBATCH --time=00:15:00
#SBATCH --output=slurm-bootstrap-%j.out
#SBATCH --error=slurm-bootstrap-%j.err

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

srun ./run-gurobi.bash python run_bootstrap.py
