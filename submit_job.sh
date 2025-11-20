#!/bin/bash
#SBATCH --account=project_2016692
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:4
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=24:00:00
#SBATCH --job-name=ipke_experiment
#SBATCH --output=logs/slurm_output_%j.txt
#SBATCH --error=logs/slurm_errors_%j.txt

# Create logs directory if it doesn't exist
mkdir -p logs

echo "Starting job on $(hostname)"

echo "Activating virtual environment..."
cd /projappl/project_2016692/IPKE
source .venv/bin/activate
echo "Pulling latest changes from git..."
# Pull the latest changes from the git repository
module load git
git pull
export CUDA_VISIBLE_DEVICES=0,1,2,3
echo "Running experiment..."
python scripts/run_prompting_experiments.py --config configs/prompting_grid.yaml --out-root logs/prompting_grid --evaluate true
echo "Job finished."