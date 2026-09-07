#!/bin/bash -l

#$ -l h_rt=2:00:00
#$ -l mem=8G
#$ -pe smp 4
#$ -N geno_pca
#$ -cwd

# Directory containing the chromosome-level PLINK genotype files.
GENO_DIR="/path/to/ADNI_WGS/geno"

# Directory for generated pruning and PCA files.
OUTPUT_DIR="./output"

mkdir -p "${OUTPUT_DIR}"

# per-chromosome MAF filtering + LD pruning (PLINK2)
module load plink/2.0alpha-git

for chr in $(seq 1 22); do
    plink2 \
        --pfile "${GENO_DIR}/ADNI_WGS.ref.chr${chr}" \
        --maf 0.01 \
        --indep-pairwise 1000 50 0.2 \
        --out "${OUTPUT_DIR}/prune_chr${chr}"

    plink2 \
        --pfile "${GENO_DIR}/ADNI_WGS.ref.chr${chr}" \
        --extract "${OUTPUT_DIR}/prune_chr${chr}.prune.in" \
        --make-bed \
        --out "${OUTPUT_DIR}/pruned_chr${chr}"
done

# merge the 22 pruned chromosome datasets (PLINK1.9)
module unload plink/2.0alpha-git
module load plink/1.90b3.40

rm -f "${OUTPUT_DIR}/merge_list.txt"

for chr in $(seq 2 22); do
    echo "${OUTPUT_DIR}/pruned_chr${chr}" >> "${OUTPUT_DIR}/merge_list.txt"
done

plink \
    --bfile "${OUTPUT_DIR}/pruned_chr1" \
    --merge-list "${OUTPUT_DIR}/merge_list.txt" \
    --make-bed \
    --out "${OUTPUT_DIR}/geno_pruned_all"

# PCA on the merged LD-pruned genotype set (PLINK2)
module unload plink/1.90b3.40
module load plink/2.0alpha-git

plink2 \
    --bfile "${OUTPUT_DIR}/geno_pruned_all" \
    --pca 100 \
    --out "${OUTPUT_DIR}/geno_pca"

echo "DONE"