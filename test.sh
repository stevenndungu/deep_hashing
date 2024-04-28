#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64GB
#SBATCH --time=50:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:v100:1

#SBATCH --job-name=Deep_learnig_Radio_Astronomy


# Clean up the module environment
#module --force purge
module --force purge

# Load the compilers

#module spider Python/3.10.8-GCCcore-12.2.0
#module load Python/3.10.8-GCCcore-12.2.0

#module spider PyTorch/1.12.1-foss-2022a-CUDA-11.7.0
#module load PyTorch/1.12.1-foss-2022a-CUDA-11.7.0

#python3 -m venv /home4/$USER/.envs/deep_hashing
source /home4/$USER/.envs/deep_hashing/bin/activate

python validation_script.py