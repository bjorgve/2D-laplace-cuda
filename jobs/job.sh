#!/bin/sh
#SBATCH --job-name=cuda-test
#SBATCH --qos=devel
#SBATCH --time=00:05:00
#SBATCH --nodes=1 --cpus-per-task=16
#SBATCH --partition=accel
#SBATCH --gpus=1
#SBATCH --mem-per-cpu=1G

ml purge
ml CMake/3.23.1-GCCcore-11.3.0
ml CUDA/12.0.0

./build/laplace-gpu.x 128 # Minumum Matrix size for GPU with current setup
./build/laplace-gpu.x 1000
./build/laplace-gpu.x 2000
./build/laplace-gpu.x 4000
