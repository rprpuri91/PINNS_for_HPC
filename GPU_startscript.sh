#!/bin/bash

# general configuration of the job
#SBATCH --job-name=TaylorGreen
#SBATCH --account=raise-ctp2
#SBATCH --mail-user=
#SBATCH --mail-type=ALL
#SBATCH --time=12:00:00
#SBATCH --output=gpu_red.out
#SBATCH --error=gpu_red.err

# configure node and process count on the CM
#SBATCH --partition=dc-gpu
#SBATCH --nodes=1
#SBATCH --cpus-per-task=32
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --exclusive

# gres options have to be disabled for deepv
#SBATCH --gres=gpu:1

# set modules
ml --force purge
ml Stages/2022 NVHPC/22.3 ParaStationMPI/5.5.0-1-mt NCCL/2.12.7-1-CUDA-11.5 cuDNN/8.3.1.22-CUDA-11.5
ml Python/3.9.6 libaio/0.3.112 HDF5/1.12.1-serial mpi-settings/CUDA

# set env
source /p/home/jusers/puri1/jureca/venv/envAI_jureca/bin/activate

# set comm
export CUDA_VISIBLE_DEVICES="0,1,2,3"
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK


#COMMAND="./Jureca/Taylor_Green_Vortex_PINN.py"
#EXEC="torchrun --nproc_per_node=$SLURM_GPUS_PER_NODE $COMMAND"

#COMMAND1="./Jureca/Taylor_Green_Vortex_PINN_noDom.py"
#EXEC1="torchrun --standalone --nnodes=1 --nproc_per_node=1 $COMMAND1"

COMMAND="./Jureca/Taylor_Green_Vortex_PINN_reduced.py"
EXEC="torchrun --nproc_per_node=$SLURM_GPUS_PER_NODE $COMMAND"

#COMMAND3="./Jureca/Taylor_Green_Vortex_PINN_noDom_reduced.py"
#EXEC3="torchrun --standalone --nnodes=1 --nproc_per_node=1 $COMMAND3"

#srun --nodes=1 --gres=gpu:1 $EXEC > test_0.out &
#srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 $EXEC1 > test_1.out &
srun --nodes=1 --gres=gpu:1 $EXEC > test_2.out &
#srun --exclusive --nodes=1 --ntasks=1 --gres=gpu:1 $EXEC3 > test_3.out &
wait

