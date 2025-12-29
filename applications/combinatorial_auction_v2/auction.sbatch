#!/bin/bash 

#SBATCH --job-name=combinatorial_auction_v2
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=00:30:00
#SBATCH --output=slurm-estimation-%j.out
#SBATCH --error=slurm-estimation-%j.err

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

srun ./run-gurobi.bash python run_estimation.py






