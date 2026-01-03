#!/bin/bash 

#SBATCH --job-name=auction_se
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=06:00:00
#SBATCH --output=slurm-auction-se-%j.out
#SBATCH --error=slurm-auction-se-%j.err

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

srun ./run-gurobi.bash python compute_se.py
