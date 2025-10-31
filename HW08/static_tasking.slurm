#!/usr/bin/env zsh
#SBATCH --partition=instruction
#SBATCH --time=00:03:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --output=static_tasking.output

cd $SLURM_SUBMIT_DIR
g++-14 -std=c++20 static_tasking.cpp -o static_tasking -I ~/ECE455/HW08/ -pthread
./static_tasking

