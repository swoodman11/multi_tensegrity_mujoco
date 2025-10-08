#!/bin/bash
#SBATCH --job-name=tensegrity_sac_gpu
#SBATCH --partition=gpu          # GPU partition name (adjust for your cluster)
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8        # Cores for data loading/preprocessing
#SBATCH --gres=gpu:1             # Request 1 GPU (specify type if needed)
#SBATCH --mem=32G                # System RAM
#SBATCH --time=48:00:00          # Max runtime: 24 hours
#SBATCH --output=logs/sac_training_%j.out
#SBATCH --error=logs/sac_training_%j.err
#SBATCH --mail-type=END,FAIL     # Email notifications
#SBATCH --mail-user=your.email@domain.com
#SBATCH --array=1-10                     # Run 10 jobs (1-10)


# Create logs directory
mkdir -p logs

# Load required modules (adjust for your cluster)
module purge
module load python/3.12
module load cuda/12.1          # Match your CUDA version
module load gcc/11.2.0         # If needed for compilation

# Activate conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate tensegrity_gnn

# Set environment variables
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Navigate to project directory
cd $SLURM_SUBMIT_DIR

# Run the training
echo "Starting SAC training on $(hostname) with GPU: $CUDA_VISIBLE_DEVICES"
echo "Job ID: $SLURM_JOB_ID"
echo "Number of CPUs: $SLURM_CPUS_PER_TASK"

python gpu_pretraining_SAC.py --cluster-mode

echo "Training completed"