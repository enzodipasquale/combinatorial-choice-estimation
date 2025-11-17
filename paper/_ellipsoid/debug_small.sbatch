#!/bin/bash
#SBATCH --job-name=debug_small
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=32
#SBATCH --cpus-per-task=1
#SBATCH --mem=20G
#SBATCH --time=00:45:00
#SBATCH --output=00_LOGS_SLURM/debug_small_%j.out
#SBATCH --error=00_LOGS_SLURM/debug_small_%j.err

set -euo pipefail

module purge

cd /scratch/ed2189/combinatorial-choice-estimation/experiments_paper

MPI_PROCS=32
SIZES_FILE=tmp_configs/debug_sizes.yaml
CONFIG_DIR=tmp_configs

declare -a experiments=(greedy supermod knapsack supermodknapsack)

EXIT_CODE=0

for EXP in "${experiments[@]}"; do
    echo "=================================================="
    echo "Starting debug run for ${EXP} at $(date)"
    echo "=================================================="
    CMD=(../experiments/run-gurobi.bash python run_experiment.py "${EXP}"
         --mpi "${MPI_PROCS}"
         --config "${CONFIG_DIR}/${EXP}_config.yaml"
         --sizes "${SIZES_FILE}")
    echo "Command: ${CMD[*]}"
    if ! "${CMD[@]}"; then
        echo "ERROR: ${EXP} run failed"
        EXIT_CODE=1
        break
    fi
    echo "Finished ${EXP} at $(date)"
done

exit "${EXIT_CODE}"

