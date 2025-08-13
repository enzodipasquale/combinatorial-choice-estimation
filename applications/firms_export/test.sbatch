#!/bin/bash 

#SBATCH --job-name=test_firms_export
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=15G
#SBATCH --time=00:01:00

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

echo "Threads $SLURM_CPUS_PER_TASK" > gurobi.env

srun ./run-gurobi.bash python run_estimation.py




