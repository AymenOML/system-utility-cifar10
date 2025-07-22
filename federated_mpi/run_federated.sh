#!/bin/bash
# Federated CIFAR-10 (MPI-based) for Cedar
# Usage: sbatch run_federated.sh

#SBATCH --job-name=fed-cifar10
#SBATCH --nodes=1                      # all tasks on 1 node
#SBATCH --ntasks=5                    # 1 server + 2 clients
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0-01:00:00
#SBATCH --output=logs/fed_cifar10_%j.out
#SBATCH --error=logs/fed_cifar10_%j.err
#SBATCH --mail-user=oumaliaymen@gmail.com
#SBATCH --mail-type=FAIL

# Load required modules
module purge
module load python/3.11 openmpi/4.1.5

# Activate pre-created virtual environment
source $HOME/venvs/fedcifar/bin/activate

# Move to your project directory
cd $HOME/scratch/system-utility-cifar10

# Ensure matplotlib uses non-GUI backend (important for Cedar)
export MPLBACKEND=Agg
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Create log directory if missing
mkdir -p logs

# Run federated training
srun --mpi=pmix python federated_mpi/mpi_main.py

# Save final plot to logs
cp federated_metrics.png logs/federated_metrics_${SLURM_JOB_ID}.png
