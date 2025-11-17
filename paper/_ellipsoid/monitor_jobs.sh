#!/bin/bash
# Quick script to monitor the large-scale experiment jobs

echo "=== Large-Scale Experiment Jobs ==="
echo ""
squeue -j 1599792,1599793,1599794,1599795 -o "%.18i %.9P %.20j %.8u %.2t %.10M %.6D %R" 2>/dev/null || echo "Jobs not in queue (may be running or completed)"
echo ""
echo "Recent log files:"
ls -lht slurm_logs/*_159979*.out 2>/dev/null | head -4
echo ""
echo "To check a specific job's progress:"
echo "  tail -f slurm_logs/greedy_1599792.out"
echo "  tail -f slurm_logs/supermod_1599793.out"
echo "  tail -f slurm_logs/knapsack_1599794.out"
echo "  tail -f slurm_logs/supermodknapsack_1599795.out"
