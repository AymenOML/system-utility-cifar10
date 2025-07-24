#!/bin/bash
# Federated CIFAR-10 (MPI-based) for Cedar
# Usage: sbatch run_federated.sh

#SBATCH --job-name=fed-cifar10
#SBATCH --nodes=16                    # 1 server + 30 clients
#SBATCH --ntasks=16
#SBATCH --ntasks-per-node=1          # 1 MPI process per node
#SBATCH --cpus-per-task=4
#SBATCH --mem=16384M                 # 16 GB per task
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

# Activate your virtual environment
source $HOME/venvs/fedcifar/bin/activate

# Go to your project directory
cd $HOME/scratch/system-utility-cifar10

# Set matplotlib to non-interactive mode
export MPLBACKEND=Agg
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Make sure logs directory exists
mkdir -p logs

# Run the federated training
srun --mpi=pmix python federated_mpi/mpi_main.py

# Save metrics plot (only executed by server node if coded properly)
cp federated_metrics.png logs/federated_metrics_${SLURM_JOB_ID}.png


