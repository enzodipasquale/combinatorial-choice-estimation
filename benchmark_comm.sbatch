#!/bin/bash

#SBATCH --job-name=comm_benchmark
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --cpus-per-task=1
#SBATCH --mem=100G
#SBATCH --time=00:20:00

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

echo "========================================="
echo "MPI Communication Benchmark"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Nodes: $SLURM_JOB_NODELIST"
echo "Total tasks: $SLURM_NTASKS"
echo "========================================="
echo ""

srun ./run-gurobi.bash python benchmark_comm_manager.py

echo ""
echo "Benchmark completed at $(date)"

