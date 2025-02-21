#!/bin/bash

args=
for i in "$@"; do 
  i="${i//\\/\\\\}"
  args="${args} \"${i//\"/\\\"}\""
done

if [[ "${args}" == "" ]]; then args="/bin/bash"; fi

if [[ -e /dev/nvidia0 ]]; then nv="--nv"; fi

if [[ -e /scratch/work/public/singularity/greene-ib-slurm-bind.sh ]]; then
  source /scratch/work/public/singularity/greene-ib-slurm-bind.sh
fi

if [[ "${SINGULARITY_CONTAINER}" != "" ]]; then
  export PATH=/share/apps/apptainer/bin:${PATH}
else
  export PATH=/share/apps/singularity/bin:${PATH}
fi

singularity exec ${nv} \
    --bind /share/apps/gurobi/gurobi.lic:/opt/gurobi/gurobi.lic \
    --overlay /scratch/ed2189/JMP/overlay-15GB-500K.ext3:ro \
    --overlay /scratch/work/public/singularity/openmpi-4.1.6-ubuntu-24.04.1.sqf:ro \
    /scratch/work/public/singularity/ubuntu-24.04.1.sif \
    /bin/bash -c "
unset -f which
if [[ -e /ext3/apps/openmpi/4.1.6/env.sh ]]; then source /ext3/apps/openmpi/4.1.6/env.sh; fi
if [[ -e /ext3/env.sh ]]; then source /ext3/env.sh; fi

# Set the license inside the container to the bound location
export GRB_LICENSE_FILE=/opt/gurobi/gurobi.lic
#export PYTHONPATH=/vast/wang/ed2189/gurobi/example:\${PYTHONPATH}
${args}
"
