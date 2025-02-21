#!/bin/bash

#SBATCH --nodes=1                 # Use 1 node
#SBATCH --ntasks-per-node=1       # 1 task per node
#SBATCH --cpus-per-task=4         # Number of CPUs per task
#SBATCH --time=1:00:00            # Time limit (hh:mm:ss)
#SBATCH --mem=16GB                # Memory limit
#SBATCH --job-name=master_run     # Job name
#SBATCH --output=master_%j.log    # Standard output and error log (%j adds the job ID)

module purge

# Set Gurobi license environment variable
export GRB_LICENSE_FILE="/share/apps/gurobi/gurobi.lic"

# Run the Python script using Singularity
singularity exec \
    --overlay /scratch/ed2189/JMP/overlay-15GB-500K.ext3:ro \
    /scratch/work/public/singularity/ubuntu-24.04.1.sif \
    /bin/bash -c "source /ext3/env.sh; python master.py"
