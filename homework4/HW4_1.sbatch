#!/bin/bash 
#SBATCH --gres=gpu:1
#SBATCH --nodes=3                        # requests 3 compute servers
#SBATCH --ntasks-per-node=2              # runs 2 tasks on each server
#SBATCH --cpus-per-task=1                # uses 1 compute core per task
#SBATCH --time=5:00:00
#SBATCH --mem=100GB
#SBATCH --job-name=HW4_1
#SBATCH --output=HW4_1.out

module purge
module load cuda/11.3.1

make

./HW4_1
