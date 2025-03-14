#!/bin/bash 

#SBATCH --job-name=test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=1
#SBATCH --mem=1G
#SBATCH --time=00:02:00

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

echo "Threads $SLURM_CPUS_PER_TASK" > gurobi.env

srun ./run-gurobi.bash python test.py




