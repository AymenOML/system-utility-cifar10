#!/bin/bash
# Federated CIFAR-10 (MPI-based) for Cedar
# Usage: sbatch run_federated.sh

#SBATCH --job-name=fed-cifar10
#SBATCH --nodes=11                          # 1 server + 10 clients, each on a node
#SBATCH --gpus-per-node=v100l:1            # 1 GPU per node
#SBATCH --ntasks-per-gpu=1                 # 1 MPI process per GPU
#SBATCH --cpus-per-task=8                  # 8 CPU cores per MPI process
#SBATCH --mem-per-cpu=2G                   # 16 GB per task (8 x 2)
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/fed_cifar10_%j.out
#SBATCH --error=logs/fed_cifar10_%j.err
#SBATCH --mail-user=oumaliaymen@gmail.com
#SBATCH --mail-type=ALL

# Load required modules
module --force purge
module load StdEnv/2023
module load python/3.11
module load openmpi/4.1.5
module load mpi4py/4.0.3

# Activate virtual environment
source $HOME/venvs/fedcifar/bin/activate

# Move to project directory
cd $HOME/scratch/system-utility-cifar10

# Set matplotlib to non-interactive mode
export MPLBACKEND=Agg
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create logs directory if not exist
mkdir -p logs

# Launch distributed training with MPI
srun --mpi=pmix python federated_mpi/mpi_main.py

# Save final plot
cp federated_metrics.png logs/federated_metrics_${SLURM_JOB_ID}.png
