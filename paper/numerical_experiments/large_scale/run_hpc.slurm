#!/bin/bash
#SBATCH --nodes=7
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=130G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/ed2189/combinatorial-choice-estimation/paper/numerical_experiments/large_scale/logs/run_hpc-%j.out
#SBATCH --error=/scratch/ed2189/combinatorial-choice-estimation/paper/numerical_experiments/large_scale/logs/err/run_hpc-%j.err
#SBATCH --account=torch_pr_531_general
module purge
module load openmpi/gcc/4.1.6
export GRB_LICENSE_FILE=/share/apps/gurobi/gurobi.lic
mpiexec --bind-to none \
    /scratch/ed2189/pyenv-openmpi-4.1.6/run-openmpi-4.1.6-ubuntu-24.04.3.bash \
    env PYTHONPATH=/scratch/ed2189/combinatorial-choice-estimation \
    python /scratch/ed2189/combinatorial-choice-estimation/paper/numerical_experiments/large_scale/run_hpc.py
