#! /bin/bash

#SBATCH --partition=h100
#SBATCH --job-name=sbatch_example
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=10
#SBATCH --time 0

torchrun --nproc_per_node=8 train_ltm.py