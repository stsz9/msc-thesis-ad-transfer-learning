import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import PCA
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
from sklearn.feature_selection import f_classif
import warnings
warnings.filterwarnings("ignore")


# labels from bench_labels.csv, covariates from the separate file for Framing 1
benchmark_data = pd.read_csv("bench_labels.csv")            # PTID, ever_AD
try:
    cov_data = pd.read_csv("benchmark_data_with_covariates.csv")  # PTID, ever_AD, AGE, ... PC6
except FileNotFoundError:
    cov_data = None

#First benchmark extension: Supervised PCA (SPCA) implementation
class SupervisedPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=10, n_screen=2000):
        self.n_components = n_components
        self.n_screen = n_screen
 
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        F, _ = f_classif(X, y)                        # univariate assoc. with the label
        F = np.nan_to_num(F, nan=0.0)
        k = min(self.n_screen, X.shape[1])
        self.selected_ = np.argsort(F)[::-1][:k]      # top-k label-associated variants
        Xsel = X[:, self.selected_]
        self.scaler_ = StandardScaler().fit(Xsel)
        Xsel = self.scaler_.transform(Xsel)
        ncomp = min(self.n_components, Xsel.shape[1], max(1, Xsel.shape[0] - 1))
        self.pca_ = PCA(n_components=ncomp, random_state=0).fit(Xsel)
        return self
 
    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xsel = X[:, self.selected_]
        Xsel = self.scaler_.transform(Xsel)
        return self.pca_.transform(Xsel)


#Second benchmark extension: Sparse PLS implementation
class PLSComponents(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=10):
        self.n_components = n_components
 
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64)
        self.scaler_ = StandardScaler().fit(X)
        Xs = self.scaler_.transform(X)
        ncomp = min(self.n_components, Xs.shape[1], max(1, Xs.shape[0] - 1))
        self.pls_ = PLSRegression(n_components=ncomp, scale=False).fit(Xs, y)
        return self
 
    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xs = self.scaler_.transform(X)
        return self.pls_.transform(Xs)

#Same evaluation used for the benchmark: cross-validation with logistic regression and AUC metric
def run_supervised_benchmark(X, y, method="pls", n_components=10, framing="alone",n_geno=None,
                             n_screen=2000, num_trials=10, name=""):
    X_arr = np.asarray(X, dtype=np.float32)
    y_arr = np.asarray(y, dtype=np.int64)
    if np.isinf(X_arr).any():
        raise ValueError("Genotype matrix contains Inf.")
 
    if method == "supervised_pca":
        rep = SupervisedPCA(n_components=n_components, n_screen=n_screen)
    elif method == "pls":
        rep = PLSComponents(n_components=n_components)
    else:
        raise ValueError("method must be 'supervised_pca' or 'pls'")
 
    if framing == "alone":
        # supervised representation only
        model = Pipeline([
            ("impute", SimpleImputer(strategy="mean")),
            ("rep", rep),
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(penalty="l2", solver="lbfgs",
                                         max_iter=5000, random_state=1)),
        ])
    elif framing == "covariates":
        geno_cols = list(range(n_geno))
        cov_cols = list(range(n_geno, X_arr.shape[1]))
        geno_branch = Pipeline([
            ("impute", SimpleImputer(strategy="mean")),
            ("rep", rep),
        ])
        ct = ColumnTransformer([
            ("geno", geno_branch, geno_cols),
            ("cov", SimpleImputer(strategy="mean"), cov_cols),
        ])
        model = Pipeline([
            ("features", ct),                 # -> [supervised components | covariates+APOE]
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(penalty="l2", solver="lbfgs",
                                         max_iter=5000, random_state=1)),
        ])
    else:
        raise ValueError("framing must be 'alone' or 'covariates'")
    pipeline_param_grid = {"model__C": [0.01, 0.03, 0.1, 0.3, 1, 3]}
    auc_scores, brier_scores, logloss_scores = [], [], []
 
    for i in range(num_trials):
        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=i)
        outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=i)
        clf = GridSearchCV(estimator=model, param_grid=pipeline_param_grid,
                           cv=inner_cv, scoring="roc_auc", n_jobs=1, error_score="raise")
        nested_score = cross_validate(clf, X=X_arr, y=y_arr, cv=outer_cv,
                                      scoring=["roc_auc", "neg_brier_score", "neg_log_loss"],
                                      n_jobs=1, error_score="raise")
        auc_scores.extend(nested_score["test_roc_auc"])
        brier_scores.extend(-nested_score["test_neg_brier_score"])
        logloss_scores.extend(-nested_score["test_neg_log_loss"])
 
    auc_scores = np.asarray(auc_scores)
    brier_scores = np.asarray(brier_scores)
    logloss_scores = np.asarray(logloss_scores)
 
    print(f"\n=== {name} ({method}, K={n_components}) ===")
    print(f"ROC-AUC:  {auc_scores.mean():.3f} (SD {auc_scores.std(ddof=1):.3f}) "
          f"[{np.percentile(auc_scores,2.5):.3f}-{np.percentile(auc_scores,97.5):.3f}]")
    print(f"Brier:    {brier_scores.mean():.3f}   (lower = better)")
    print(f"Log-loss: {logloss_scores.mean():.3f}   (lower = better)")
    return {"auc": auc_scores, "brier": brier_scores, "logloss": logloss_scores}

if __name__ == "__main__":
 
    # load the raw genotype data 
    RAW_FILE = "adni_all_raw.raw"    
    header = pd.read_csv(RAW_FILE, sep=r"\s+", nrows=0).columns.tolist()
    variant_cols = header[6:]

    dtype_map = {c: np.float32 for c in variant_cols}
    raw = pd.read_csv(RAW_FILE, sep=r"\s+", dtype=dtype_map)
    raw["IID"] = (raw["IID"].astype(str)
                  .str.replace("_s_", "_S_", regex=False)
                  .str.upper().str.strip())
    variant_cols = list(raw.columns[6:])            # cols after FID IID PAT MAT SEX PHENOTYPE
    print(f"Loaded chr genotypes: {raw.shape[0]} individuals x {len(variant_cols)} variants")
    print(raw.shape)   
    geno = raw[["IID"] + variant_cols].rename(columns={"IID": "PTID"})

    # covariate columns for Framing 1 (from benchmark_data_with_covariates.csv)
    COV_COLS = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4_COUNT",
                "PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]
    have_cov = (cov_data is not None) and all(c in cov_data.columns for c in COV_COLS)
    if have_cov:
        labels_cov = cov_data[["PTID", "ever_AD"] + COV_COLS].copy()
    else:
        labels_cov = benchmark_data[["PTID", "ever_AD"]].copy()
    labels_cov["PTID"] = labels_cov["PTID"].astype(str).str.upper().str.strip()
    merged = labels_cov.merge(geno, on="PTID", how="inner")
    print(f"Merged: n={len(merged)} (cases={int(merged['ever_AD'].sum())}, "
          f"controls={int((merged['ever_AD']==0).sum())})  covariates_present={have_cov}", flush=True)
 
    yg = merged["ever_AD"].to_numpy(dtype=np.int64)
    Xg = merged[variant_cols].to_numpy(dtype=np.float32) 


    print("\nFRAMING 2: representation ALONE", flush=True)
    for method in ["pls", "supervised_pca"]:
        run_supervised_benchmark(Xg, yg, method=method, n_components=10,
                                 n_screen=2000, num_trials=10, framing="alone",
                                 name=f"{method} alone K=10")
 
    if have_cov:
        print("\nFRAMING 1: representation + covariates + APOE ", flush=True)
        Xcov = merged[COV_COLS].to_numpy(dtype=np.float32)
        Xcomb = np.hstack([Xg, Xcov])                          
        n_geno = Xg.shape[1]
        for method in ["pls", "supervised_pca"]:
            run_supervised_benchmark(Xcomb, yg, method=method, n_components=10,
                                     n_screen=2000, num_trials=10, framing="covariates",
                                     n_geno=n_geno, name=f"{method} + covariates + APOE K=10")
    else:
        print("\n[covariates not found in bench_labels.csv -> skipping Framing 1]", flush=True)
        print("To run Framing 1, re-export bench_labels.csv from pgs.py with columns:", flush=True)
        print("  PTID, ever_AD, AGE, PTGENDER, PTEDUCAT, APOE4_COUNT, PC1..PC6", flush=True)