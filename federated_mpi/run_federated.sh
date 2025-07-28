#!/bin/bash
#SBATCH --job-name=fed-cifar10
#SBATCH --nodes=51                      # 1 server + 50 clients
#SBATCH --ntasks=51                     # 1 MPI process per node
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1              # Request 1 GPU per node (A100 on Narval)
#SBATCH --cpus-per-task=8              # 8 CPU cores per process
#SBATCH --mem-per-cpu=4G               # 32 GB per process (8 x 4)
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/fed_cifar10_%j.out
#SBATCH --error=logs/fed_cifar10_%j.err
#SBATCH --mail-user=oumaliaymen@gmail.com
#SBATCH --mail-type=ALL

# --- IMPORTANT CHANGE FOR NARVAL ---
# You MUST specify your allocation account. Replace 'def-someprof' with your professor's account.
#SBATCH --account=def-cherkaou-ab

# Load required modules (these are generally consistent across clusters)
module --force purge
module load StdEnv/2023
module load python/3.11
module load openmpi/4.1.5
module load mpi4py/4.0.3

# Activate virtual environment
source $HOME/venvs/fedcifar/bin/activate

# Move to project directory
cd $HOME/scratch/system-utility-cifar10

# Ensure matplotlib uses non-GUI backend
export MPLBACKEND=Agg
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- REMOVED FOR NARVAL ---
# The PSM2_CUDA variable is specific to the interconnect on some Cedar nodes
# and is not needed on Narval.
# export PSM2_CUDA=0

# Create logs directory
mkdir -p logs

# Launch training
srun --mpi=pmix python federated_mpi/mpi_main.py

# Copy metrics plot if it exists
[ -f federated_metrics.png ] && cp federated_metrics.png logs/federated_metrics_${SLURM_JOB_ID}.png
