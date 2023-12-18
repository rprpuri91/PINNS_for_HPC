#!/bin/bash

# general configuration of the job
#SBATCH --job-name=TGV
#SBATCH --account=raise-ctp2
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --output=job_tgv0.out
#SBATCH --error=job_tgv0.err
#SBATCH --time=02:00:00

# configure node and process count on the CM
#SBATCH --partition=dc-gpu-devel
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=128
#SBATCH --gpus-per-node=4
#SBATCH --exclusive

# gres options have to be disabled for deepv
#SBATCH --gres=gpu:1

# parameters
debug=false # do debug
bs=8       # batch-size
epochs=30    # epochs
lr=0.003     # learning rate
gamma=0.999 # gamma for decay
restartInt=2000 # restart interval for saving
train_percent=0 # ground truth percentage in train
test_ID='0' # test ID based on training data 
NN='PINN' # training model type

# AT
#dataDir="/p/scratch/raise-ctp2/T31_LD/"
#COMMAND="./Jureca/Taylor_Green_Vortex_PINN_noCentre.py"
#COMMAND="./Jureca/Taylor_Green_Vortex_Continuum_data_variation.py"
COMMAND="./Jureca/Taylor_Green_Vortex_Continuum_data_free.py"
#COMMAND="./data/data_Taylor_Green_Vortex_PINN.py"
EXEC="$COMMAND \
  --batch-size $bs \
  --epochs $epochs \
  --lr $lr \
  --restart-int $restartInt\
  --train_percent $train_percent\
  --test_ID $test_ID\
  --model_type $NN\
  --nworker $SLURM_CPUS_PER_TASK"
#--data-dir $dataDir"


### do not modify below ###


# set modules
ml --force purge
ml Stages/2023 StdEnv/2023 NVHPC/23.1 OpenMPI/4.1.4 cuDNN/8.6.0.163-CUDA-11.7
ml Python/3.10.4 CMake HDF5 PnetCDF libaio/0.3.112
 

#ml Stages/2023 StdEnv/2023 NVHPC/23.1 OpenMPI/4.1.4 cuDNN/8.6.0.163-CUDA-11.7 Python/3.10.4 HDF5 libaio/0.3.112

# set env
source /p/home/jusers/puri1/jureca/scratch/raise-ctp2/puri1/venv/envAI_jureca/bin/activate
#source /p/project/prcoe12/RAISE/envAI_jureca/bin/activate
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
