#!/bin/bash 

#SBATCH --job-name=greedy_experiment
#SBATCH --nodes=5
#SBATCH --ntasks-per-node=5
#SBATCH --cpus-per-task=1
#SBATCH --mem=10G
#SBATCH --time=02:00:00

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

echo "Starting Greedy Experiment with $SLURM_NNODES nodes and $SLURM_NTASKS_PER_NODE tasks per node"

srun ./run-gurobi.bash python ../../experiments/experiment_inversion_greedy.py
