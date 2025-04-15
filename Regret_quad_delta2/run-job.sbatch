#!/bin/bash 

#SBATCH --job-name=Regret_quad_delta_2
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=25G
#SBATCH --time=24:00:00

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

echo "Threads $SLURM_CPUS_PER_TASK" > gurobi.env

srun ./run-gurobi.bash python job-main.py




