#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=dependent_async.output

cd $SLURM_SUBMIT_DIR
g++-14 -std=c++20 dependent_async.cpp -o dependent_async -I ~/ECE455/HW08/ -pthread
./dependent_async

