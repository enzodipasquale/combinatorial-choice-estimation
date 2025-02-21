#!/bin/bash

# Loop to submit pricing and master jobs N = 300 times, I put 2
for iter in {1..2}; do
    echo "Submitting pricing job #$iter"
    
    # Wait for the queue to be empty before submitting the pricing job
    while [ $(squeue -u $USER | grep -c "pricing") -gt 0 ]; do
        echo "Waiting for pricing jobs to finish..."
        sleep 10  # Check every 10 seconds
    done
    
    # Submit pricing job
    sbatch pricing.sbatch
    echo "Pricing job #$iter submitted."
    
    # Wait for the pricing job to complete
    while [ $(squeue -u $USER | grep -c "pricing") -gt 0 ]; do
        echo "Waiting for pricing job #$iter to complete..."
        sleep 10
    done
    
    echo "Submitting master job #$iter"
    
    # Submit master job
    sbatch master.sbatch
    echo "Master job #$iter submitted."
    
    # Wait for the master job to complete
    while [ $(squeue -u $USER | grep -c "master") -gt 0 ]; do
        echo "Waiting for master job #$iter to complete..."
        sleep 10
    done
done

echo "All jobs have been submitted."
