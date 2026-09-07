#!/bin/bash -l

#$ -l h_rt=2:00:00
#$ -l mem=8G
#$ -pe smp 2
#$ -N missing_check
#$ -cwd

module load plink/2.0alpha-git

# Directory containing the chromosome-level PLINK genotype files.
GENO_DIR="/path/to/ADNI_WGS/geno"

# Directory for generated missingness results.
OUTPUT_DIR="./missingness_output"

mkdir -p "${OUTPUT_DIR}"

# Calculate sample-level missingness for each chromosome.
for chr in $(seq 1 22); do
    plink2 \
        --pfile "${GENO_DIR}/ADNI_WGS.ref.chr${chr}" \
        --missing sample-only \
        --out "${OUTPUT_DIR}/miss_chr${chr}"
done

echo "DONE"