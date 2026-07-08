#!/bin/bash
#SBATCH --job-name=dsprites_exp
#SBATCH --partition=hpc-high
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --array=0-5
#SBATCH --output=/mnt/home/amitra/Deep_Dive/logs/dsprites_%A_%a.out
#SBATCH --error=/mnt/home/amitra/Deep_Dive/logs/dsprites_%A_%a.err

source /opt/conda/bin/activate base

pip install -q seaborn wandb ipython

# Map SLURM array task ID (0–5) to depth (3–8)
DEPTHS=(3 4 5 6 7 8)
DEPTH=${DEPTHS[$SLURM_ARRAY_TASK_ID]}

echo "Running depth=${DEPTH} on $(hostname), GPU=${CUDA_VISIBLE_DEVICES}"

cd /mnt/home/amitra/Deep_Dive
python data/toy/dsprites/train_dsprites.py --depth ${DEPTH}

# After all 6 jobs finish, generate summary plots with:
#   python data/toy/dsprites/train_dsprites.py --summary
