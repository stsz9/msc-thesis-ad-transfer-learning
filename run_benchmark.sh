#!/bin/bash -l

#$ -cwd
#$ -N sup_bench_full
#$ -l h_rt=10:00:00
#$ -l mem=16G
#$ -pe smp 4
#$ -l tmpfs=10G
#$ -o sup_bench_full.out
#$ -e sup_bench_full.err

# Limit numerical libraries to a single thread to avoid
# oversubscription during cross-validation.
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export VECLIB_MAXIMUM_THREADS=1

echo "Job started:"
date

echo "Running on:"
hostname

echo "Python:"
python --version

# Run the supervised raw-genotype benchmark.
python supervised_benchmark.py

echo "Job finished:"
date