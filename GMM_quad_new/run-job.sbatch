#!/bin/bash 

#SBATCH --job-name=GMM_quad-new
#SBATCH --nodes=15
#SBATCH --ntasks-per-node=20
#SBATCH --cpus-per-task=1
#SBATCH --mem=25G
#SBATCH --time=03:30:00

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

echo "Threads $SLURM_CPUS_PER_TASK" > gurobi.env

srun ./run-gurobi.bash python job-main.py




