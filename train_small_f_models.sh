#!/bin/bash

#BATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --output=train_ftiny.out

python train.py --model_size tiny --dataset rfam_f90 > rfam_f90_tiny.log
