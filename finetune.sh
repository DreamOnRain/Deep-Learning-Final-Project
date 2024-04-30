#!/bin/bash

#SBATCH --time=2-10:30:00
#SBATCH --job-name=cs5242_project
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=jiyuyu@comp.nus.edu.sg
#SBATCH --gpus=a100-80:1
#SBATCH --partition=long

python finetune/finetune.py
python merge.py
