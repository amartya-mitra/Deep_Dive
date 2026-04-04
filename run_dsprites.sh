#!/bin/bash
#SBATCH --job-name=dsprites_exp
#SBATCH --partition=hpc-mid
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --output=/mnt/home/amitra/Deep_Dive/logs/dsprites_%j.out
#SBATCH --error=/mnt/home/amitra/Deep_Dive/logs/dsprites_%j.err

source /opt/conda/bin/activate base

pip install -q seaborn wandb ipython

cd /mnt/home/amitra/Deep_Dive
python data/toy/dsprites/train_dsprites.py
