#!/bin/bash 

#SBATCH --job-name=auction_estimation
#SBATCH --nodes=35
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=80G
#SBATCH --time=03:30:00
#SBATCH --output=slurm-estimation-%j.out

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

srun ./run-gurobi.bash python run_estimation.py







