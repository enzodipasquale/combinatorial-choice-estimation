#!/bin/bash

#SBATCH --array=1-255       # Create a job array of 255 tasks
#SBATCH --nodes=1              # Use 1 node
#SBATCH --ntasks-per-node=1    # 1 task per node
#SBATCH --cpus-per-task=1      # 1 CPU per task
#SBATCH --time=1:00:00         # Time limit for each task
#SBATCH --mem=1GB          # Memory per task
#SBATCH --job-name=pricing # Name of the job array
#SBATCH --output=/dev/null  # Disable standard output (no .out file)

module purge

export GRB_LICENSE_FILE="/share/apps/gurobi/gurobi.lic"

# Run the Python script for each task in the array
singularity exec \
    --overlay /scratch/ed2189/JMP/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/ubuntu-24.04.1.sif \
    /bin/bash -c "source /ext3/env.sh; python pricing.py $SLURM_ARRAY_TASK_ID"
