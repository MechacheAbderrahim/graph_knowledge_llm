#!/bin/bash
#SBATCH --job-name="KG"
#SBATCH --output="outputs/KG.out"
#SBATCH --error="outputs/KG.err"
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --hint=nomultithread
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --qos=qos_gpu-t3
#SBATCH -A acu@v100
#SBATCH -C v100-32g

module purge
module load pytorch-gpu/py3/2.4.0

python test_local_model_load.py
