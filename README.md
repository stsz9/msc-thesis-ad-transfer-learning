Zero-Shot Bayesian Transfer Learning for Alzheimer's Disease Prediction

The use of genetic data in machine learning applications is an example of a high dimensional data
challenge where the number of parameters exceeds that of the target patient cohort (p ≫ n), result-
ing in variance inflation, overfitting, and poor generalisation. This thesis addresses this challenge
by constructing a low-dimensional representation of the genetic data, using Alzheimer’s disease
as the application domain. The framework is constructed in two separate stages: The first stage
learns a low-dimensional genetic representation externally from genome-wide association stud-
ies (GWAS, n ∼ 105–106), and constructs polygenic scores using a Bayesian shrinkage method,
SBayesRC. This representation is considered zero-shot because the learning was performed inde-
pendently of the target cohort labels. The second stage fits a machine learning model for disease
prediction. The results were then benchmarked against a representation learned directly from
raw-genotype data across a range of dimensionalities. The p ≫ n problem is demonstrated by
the directly learned representation’s performance degradation as dimensionality increases. 

This repository contains the code used to evaluate externally derived polygenic score (PGS) representations and raw-genotype representations for Alzheimer's disease prediction.

The analyses include:

- Construction and validation of polygenic score features
- Alzheimer's disease prediction using demographic, genetic, and PGS predictors
- Comparison of single-trait and multi-trait PGS representations
- Raw-genotype benchmark models using PCA, supervised PCA, and PLS
- Repeated nested cross-validation and bootstrap optimism correction
- Genotype quality-control and preprocessing scripts

- ## Repository Structure

### Main analysis

- `pgs.py`  

  Main PGS analysis and predictive modelling pipeline.

- `outlier_detection.py`  

  Genetic principal-component outlier detection.

- `supervised_benchmark.py`  

  Raw-genotype supervised representation benchmark.
### Genotype preprocessing

- `run_genopca.sh`  

  LD pruning and genotype PCA pipeline.

- `run_missing.sh`  

  Sample-level genotype missingness analysis for each chromosome.
  ### Benchmark execution

- `run_benchmark.sh`  

  Job script for running the target-supervised genetic representation benchmark.
  ## Data

The analyses use participant-level data from the Alzheimer's Disease Neuroimaging Initiative (ADNI).
Participant-level phenotype and genotype data are **not included in this repository** because access 
is subject to ADNI data-use requirements.

External GWAS summary statistics were used to construct polygenic scores.
## Modelling

Predictive models were evaluated using repeated nested cross-validation.

The primary comparisons evaluate:

1. Baseline demographic covariates
2. Genetic ancestry principal components
3. APOE ε4 allele count
4. Alzheimer's disease PGS
5. Multi-trait PGS representations
6. Raw-genotype representations

Raw-genotype benchmark methods include unsupervised PCA, supervised PCA,
and partial least squares (PLS).
## Reproducibility

Python scripts were run using Python 3 with packages including:

- NumPy

- pandas

- scikit-learn

- SciPy

- statsmodels

- matplotlib

Genotype preprocessing was performed using PLINK/PLINK2.
