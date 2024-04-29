#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=50:00:00
#SBATCH --partition=gpu
#SBATCH --gpus-per-node=a100

#SBATCH --job-name=deep_hashing

module --force purge
module load Python/3.10.8-GCCcore-12.2.0

#python3 -m venv /home4/$USER/.envs/deep_hashing
source /home4/$USER/.envs/deep_hashing/bin/activate

python validation_script.py
#pip install fastai==2.7.10
#squeue -u $USER