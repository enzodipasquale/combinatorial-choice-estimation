#!/bin/bash 

#SBATCH --job-name=val_auction_small
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=1:00:00
#SBATCH --output=slurm_%j.out
#SBATCH --error=slurm_%j.err

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

cd /scratch/ed2189/combinatorial-choice-estimation/validation_auction

srun ../applications/combinatorial_auction/run-gurobi.bash python run_validation.py --config config_small.yaml

