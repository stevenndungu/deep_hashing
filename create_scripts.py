import os

script_content = """#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=200:00:00
#SBATCH --partition=regular
#SBATCH --job-name=dht2_{0}
#SBATCH --output=dht2_{0}.out
#SBATCH --mem=64GB

# Clean up the module environment
#module --force purge
#module purge

# Load the compilers
module load Python/3.10.8-GCCcore-12.2.0

#python3 -m venv /home4/$USER/.envs/deep_hashing
source /home4/$USER/.envs/deep_hashing/bin/activate
python Image_retrieval_script13082024.py --num {1}

"""
#          python create_scripts.py

for num in range(1, 27):
    script_name = f"run_{num}.sh"
    with open(script_name, "w") as script_file:
        script_file.write(script_content.format(num, num))
        print(f"Created {script_name}")

print("Done!")
