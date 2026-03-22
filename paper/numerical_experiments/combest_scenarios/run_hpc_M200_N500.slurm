#!/bin/bash
#SBATCH --nodes=7
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=60G
#SBATCH --time=48:00:00
#SBATCH --output=/scratch/ed2189/combinatorial-choice-estimation/paper/numerical_experiments/combest_scenarios/logs/run_hpc_M200_N500-%j.out
#SBATCH --error=/scratch/ed2189/combinatorial-choice-estimation/paper/numerical_experiments/combest_scenarios/logs/err/run_hpc_M200_N500-%j.err
#SBATCH --account=torch_pr_531_general

# M=200 only, N=500: 224 ranks, 60G RAM

module purge
module load openmpi/gcc/4.1.6
export GRB_LICENSE_FILE=/share/apps/gurobi/gurobi.lic

PROJ=/scratch/ed2189/combinatorial-choice-estimation

mkdir -p ${PROJ}/paper/numerical_experiments/combest_scenarios/logs/err
mkdir -p ${PROJ}/paper/numerical_experiments/combest_scenarios/results/raw
mkdir -p ${PROJ}/paper/numerical_experiments/combest_scenarios/results/hpc

mpiexec --bind-to none /scratch/ed2189/pyenv-openmpi-4.1.6/run-openmpi-4.1.6-ubuntu-24.04.3.bash env PYTHONPATH=${PROJ} python ${PROJ}/paper/numerical_experiments/combest_scenarios/run_hpc.py --N 500 --M 200
