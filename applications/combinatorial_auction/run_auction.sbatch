#!/bin/bash 

#SBATCH --job-name=combinatorial_auction
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=20
#SBATCH --mem=40G
#SBATCH --time=07:00:00

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

# echo "Threads $SLURM_CPUS_PER_TASK" > gurobi.env

srun ./run-gurobi.bash python run_estimation.py




