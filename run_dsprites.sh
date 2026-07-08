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

# Encoder mode toggle — propagate from the submitting shell, default flat_pixels.
# Submit both diagnostics in parallel with:
#   ENCODER_MODE=flat_pixels sbatch run_dsprites.sh
#   ENCODER_MODE=frozen_cnn  sbatch run_dsprites.sh
export ENCODER_MODE=${ENCODER_MODE:-flat_pixels}

# Map SLURM array task ID (0–5) to depth (3–8)
DEPTHS=(3 4 5 6 7 8)
DEPTH=${DEPTHS[$SLURM_ARRAY_TASK_ID]}

echo "Running ENCODER_MODE=${ENCODER_MODE} depth=${DEPTH} on $(hostname), GPU=${CUDA_VISIBLE_DEVICES}"

cd /mnt/home/amitra/Deep_Dive
python data/toy/dsprites/train_dsprites.py --depth ${DEPTH}

# After all 6 jobs for a mode finish, generate that mode's summary plots with:
#   ENCODER_MODE=flat_pixels python data/toy/dsprites/train_dsprites.py --summary
#   ENCODER_MODE=frozen_cnn  python data/toy/dsprites/train_dsprites.py --summary
# Then compile the cross-encoder comparison:
#   python data/toy/dsprites/compile_encoder_comparison.py
