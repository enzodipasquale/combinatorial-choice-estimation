#!/bin/bash
# Submit all experiment jobs to SLURM

cd /scratch/ed2189/combinatorial-choice-estimation/experiments_paper

echo "Submitting all experiment jobs..."

JOB1=$(sbatch run_greedy.sbatch | awk '{print $4}')
echo "Submitted greedy job: $JOB1"

JOB2=$(sbatch run_supermod.sbatch | awk '{print $4}')
echo "Submitted supermod job: $JOB2"

JOB3=$(sbatch run_knapsack.sbatch | awk '{print $4}')
echo "Submitted knapsack job: $JOB3"

JOB4=$(sbatch run_supermodknapsack.sbatch | awk '{print $4}')
echo "Submitted supermodknapsack job: $JOB4"

echo ""
echo "All jobs submitted!"
echo "Job IDs:"
echo "  greedy: $JOB1"
echo "  supermod: $JOB2"
echo "  knapsack: $JOB3"
echo "  supermodknapsack: $JOB4"
echo ""
echo "Monitor with: squeue -u $USER"

