#!/bin/bash -l

#$ -cwd
#$ -N sup_pls
#$ -l h_rt=08:00:00
#$ -l mem=16G
#$ -pe smp 4
#$ -l tmpfs=10G
#$ -o sup_pls.out
#$ -e sup_pls.err

# Limit numerical libraries to a single thread.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "Job started:"
date

# Run the PLS benchmark.
python run_pls.py

echo "Job finished:"
date