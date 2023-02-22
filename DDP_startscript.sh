#!/bin/bash

# general configuration of the job
#SBATCH --job-name=TorchTest
#SBATCH --account=raise-ctp2
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job_redNC.out
#SBATCH --error=job_redNC.err
#SBATCH --time=06:00:00

# configure node and process count on the CM
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=32
#SBATCH --gpus-per-node=2
#SBATCH --exclusive

# gres options have to be disabled for deepv
#SBATCH --gres=gpu:2

# parameters
debug=false # do debug
bs=8       # batch-size
epochs=10000    # epochs
lr=0.001     # learning rate

# AT
#dataDir="/p/scratch/raise-ctp2/T31_LD/"
COMMAND="./Jureca/Taylor_Green_Vortex_PINN_noCentre.py"
EXEC="$COMMAND \
  --batch-size $bs \
  --epochs $epochs \
  --lr $lr \
  --nworker $SLURM_CPUS_PER_TASK"
#--data-dir $dataDir"


### do not modify below ###


# set modules
ml --force purge
ml Stages/2022 NVHPC/22.3 ParaStationMPI/5.5.0-1-mt NCCL/2.12.7-1-CUDA-11.5 cuDNN/8.3.1.22-CUDA-11.5 
ml Python/3.9.6 libaio/0.3.112 HDF5/1.12.1-serial mpi-settings/CUDA 

# set env
source /p/home/jusers/puri1/jureca/venv/envAI_jureca/bin/activate

# sleep a sec
sleep 1

# job info 
echo "DEBUG: TIME: $(date)"
echo "DEBUG: EXECUTE: $EXEC"
echo "DEBUG: SLURM_SUBMIT_DIR: $SLURM_SUBMIT_DIR"
echo "DEBUG: SLURM_JOB_ID: $SLURM_JOB_ID"
echo "DEBUG: SLURM_JOB_NODELIST: $SLURM_JOB_NODELIST"
echo "DEBUG: SLURM_NNODES: $SLURM_NNODES"
echo "DEBUG: SLURM_NTASKS: $SLURM_NTASKS"
echo "DEBUG: SLURM_TASKS_PER_NODE: $SLURM_TASKS_PER_NODE"
echo "DEBUG: SLURM_SUBMIT_HOST: $SLURM_SUBMIT_HOST"
echo "DEBUG: SLURMD_NODENAME: $SLURMD_NODENAME"
echo "DEBUG: CUDA_VISIBLE_DEVICES: $CUDA_VISIBLE_DEVICES"
if [ "$debug" = true ] ; then
  export NCCL_DEBUG=INFO
fi
echo

# set comm
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OMP_NUM_THREADS=1
if [ "$SLURM_CPUS_PER_TASK" > 0 ] ; then
  export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
fi

# launch
srun --cpu-bind=none bash -c "torchrun \
    --log_dir='logs' \
    --nnodes=$SLURM_NNODES \
    --nproc_per_node=$SLURM_GPUS_PER_NODE \
    --rdzv_id=$SLURM_JOB_ID \
    --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
    --rdzv_backend=c10d \
    --rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)'i:29500 \
    $EXEC"

# add --globres=fs:cscratch@just flag to l. 78 if High Performance Storage Tier (HPST)

# nsys profiler: following https://gist.github.com/mcarilli/376821aa1a7182dfcf59928a7cde3223
#srun --cpu-bind=none nsys profile \
#        --trace=cublas,cuda,cudnn,nvtx,osrt \
#        --sample=cpu \
#        --stats=true \
#        --force-overwrite=true \
#        -o ./prof.out bash -c "torchrun \
#        --log_dir='logs' \
#        --nnodes=$SLURM_NNODES \
#        --nproc_per_node=$SLURM_GPUS_PER_NODE \
#        --rdzv_id=$SLURM_JOB_ID \
#        --rdzv_conf=is_host=\$(((SLURM_NODEID)) && echo 0 || echo 1) \
#        --rdzv_backend=c10d \
#        --rdzv_endpoint='$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)'i:29500 \
#        $EXEC"

# eof
