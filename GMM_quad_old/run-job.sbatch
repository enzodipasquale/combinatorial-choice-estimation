#!/bin/bash 

#SBATCH --job-name=GMM_quad-old
#SBATCH --nodes=20
#SBATCH --ntasks-per-node=15 
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=00:06:00

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

echo "Threads $SLURM_CPUS_PER_TASK" > gurobi.env

srun ./run-gurobi.bash python job-main.py




