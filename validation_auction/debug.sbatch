#!/bin/bash 

#SBATCH --job-name=val_debug
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=1
#SBATCH --mem=40G
#SBATCH --time=2:00:00
#SBATCH --output=debug_%j.out
#SBATCH --error=debug_%j.err

module purge

export GRB_LICENSE_FILE="/home/ed2189/gurobi.lic"

cd /scratch/ed2189/combinatorial-choice-estimation/validation_auction

echo "=========================================="
echo "Debug: Testing Step 1 - Extract Parameters"
echo "=========================================="
if [ -f theta_hat_real.npy ]; then
    echo "theta_hat_real.npy already exists, skipping Step 1"
    echo "To rerun Step 1, delete theta_hat_real.npy first"
else
    srun ../applications/combinatorial_auction/run-gurobi.bash mpirun -n 4 python extract_parameters.py
    
    if [ $? -ne 0 ]; then
        echo "ERROR: Step 1 failed"
        exit 1
    fi
fi

echo ""
echo "=========================================="
echo "Debug: Testing Step 2 - Generate Synthetic Data"
echo "=========================================="
srun ../applications/combinatorial_auction/run-gurobi.bash mpirun -n 4 python generate_synthetic_data.py --config config_small.yaml --seed 42

if [ $? -ne 0 ]; then
    echo "ERROR: Step 2 failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Debug: Testing Step 3 - Run Validation"
echo "=========================================="
srun ../applications/combinatorial_auction/run-gurobi.bash mpirun -n 4 python run_validation.py --config config_small.yaml --replication-id 0

if [ $? -ne 0 ]; then
    echo "ERROR: Step 3 failed"
    exit 1
fi

echo ""
echo "=========================================="
echo "Debug: All steps completed successfully!"
echo "=========================================="

