#!/bin/bash
#SBATCH --job-name=fed-cifar10
#SBATCH --nodes=11                     # 1 server + 10 clients
#SBATCH --ntasks=11                    # 1 MPI process per node
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1              # 1 GPU per node
#SBATCH --cpus-per-task=8              # 8 CPU cores per process
#SBATCH --mem-per-cpu=4G               # 32 GB / task (8 x 4)
#SBATCH --time=0-12:00:00
#SBATCH --output=logs/fed_cifar10_%j.out
#SBATCH --error=logs/fed_cifar10_%j.err
#SBATCH --mail-user=oumaliaymen@gmail.com
#SBATCH --mail-type=ALL
#SBATCH --account=def-cherkaou-ab

# --- Modules (Narval) ---
module --force purge
module load StdEnv/2023
module load python/3.11
module load openmpi/4.1.5
module load mpi4py
# If your TF needs CUDA/CuDNN, load them here as you normally do.

# --- venv ---
source "$HOME/venvs/fedcifar/bin/activate"

# --- Always start from the submission directory ---
cd "$SLURM_SUBMIT_DIR"

# --- Env for plotting & threads ---
export MPLBACKEND=Agg
export OMP_NUM_THREADS="$SLURM_CPUS_PER_TASK"

# --- PYTHONPATH (so imports resolve regardless of where the job was submitted) ---
export PYTHONPATH="$PWD:$PWD/data_treatment/Code/Normalization:$PWD/data_treatment/Code/Noramlization:$PWD/data_treatment/Code/Labelling:$PYTHONPATH"

# --- Enforce MLP selection + threshold mode ---
export FEDSEL_ENFORCE=1
export FEDSEL_THRESH=0.50
export FEDSEL_JOURNAL_SELECTED_ONLY=1   # << only selected clients are saved in per-round CSVs


# --- Nice to have: ensure logs and data dirs exist ---
mkdir -p logs Data/logs Model

# (Optional) quick smoke test: show where we are and that model exists
echo "[job] CWD=$PWD"
if [ -f Model/mlp_client_selector.h5 ]; then
  echo "[job] Found Model/mlp_client_selector.h5"
else
  echo "[job][WARN] Model/mlp_client_selector.h5 NOT FOUND under $PWD/Model"
fi

# --- Launch training (unbuffered python; pmix or pmix_v3 depending on Narval) ---
srun --mpi=pmix_v3 python -u federated_mpi/mpi_main.py
