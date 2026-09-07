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

benchmark_data = pd.read_csv("bench_labels.csv")                    # PTID, ever_AD
try:
    cov_data = pd.read_csv("benchmark_data_with_covariates.csv")   # PTID, ever_AD, AGE ... PC6
except FileNotFoundError:
    cov_data = None


# Supervised PCA: screen variants by association with y, then PCA on the subset.
class SupervisedPCA(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=10, n_screen=2000):
        self.n_components = n_components
        self.n_screen = n_screen
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y)
        F, _ = f_classif(X, y); F = np.nan_to_num(F, nan=0.0)
        k = min(self.n_screen, X.shape[1])
        self.selected_ = np.argsort(F)[::-1][:k]
        Xsel = X[:, self.selected_]
        self.scaler_ = StandardScaler().fit(Xsel); Xsel = self.scaler_.transform(Xsel)
        ncomp = min(self.n_components, Xsel.shape[1], max(1, Xsel.shape[0] - 1))
        self.pca_ = PCA(n_components=ncomp, random_state=0).fit(Xsel)
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xsel = self.scaler_.transform(X[:, self.selected_])
        return self.pca_.transform(Xsel)


# Partial least squares (PLS): components maximising covariance with y.
class PLSComponents(BaseEstimator, TransformerMixin):
    def __init__(self, n_components=10):
        self.n_components = n_components
    def fit(self, X, y):
        X = np.asarray(X, dtype=np.float32); y = np.asarray(y, dtype=np.float64)
        self.scaler_ = StandardScaler().fit(X); Xs = self.scaler_.transform(X)
        ncomp = min(self.n_components, Xs.shape[1], max(1, Xs.shape[0] - 1))
        self.pls_ = PLSRegression(n_components=ncomp, scale=False).fit(Xs, y)
        return self
    def transform(self, X):
        X = np.asarray(X, dtype=np.float32)
        Xs = self.scaler_.transform(X)
        return self.pls_.transform(Xs)


def run_supervised_benchmark(X, y, method="pls", n_components=10, framing="alone",
                             n_geno=None, n_screen=2000, num_trials=5, name=""):
    X_arr = np.asarray(X, dtype=np.float32); y_arr = np.asarray(y, dtype=np.int64)
    if np.isinf(X_arr).any():
        raise ValueError("Genotype matrix contains Inf.")
    rep = SupervisedPCA(n_components, n_screen) if method == "supervised_pca" else PLSComponents(n_components)

    if framing == "alone":
        model = Pipeline([("impute", SimpleImputer(strategy="mean")), ("rep", rep),
                          ("scaler", StandardScaler()),
                          ("model", LogisticRegression(penalty="l2", solver="lbfgs",
                                                       max_iter=5000, random_state=1))])
    else:  # covariates
        geno_cols = list(range(n_geno)); cov_cols = list(range(n_geno, X_arr.shape[1]))
        geno_branch = Pipeline([("impute", SimpleImputer(strategy="mean")), ("rep", rep)])
        ct = ColumnTransformer([("geno", geno_branch, geno_cols),
                                ("cov", SimpleImputer(strategy="mean"), cov_cols)])
        model = Pipeline([("features", ct), ("scaler", StandardScaler()),
                          ("model", LogisticRegression(penalty="l2", solver="lbfgs",
                                                       max_iter=5000, random_state=1))])

    grid = {"model__C": [0.01, 0.03, 0.1, 0.3, 1, 3]}
    aucs, briers, lls = [], [], []
    for i in range(num_trials):
        inner = StratifiedKFold(4, shuffle=True, random_state=i)
        outer = StratifiedKFold(4, shuffle=True, random_state=i)
        clf = GridSearchCV(model, grid, cv=inner, scoring="roc_auc", n_jobs=1, error_score="raise")
        sc = cross_validate(clf, X_arr, y_arr, cv=outer,
                            scoring=["roc_auc", "neg_brier_score", "neg_log_loss"],
                            n_jobs=1, error_score="raise")
        aucs.extend(sc["test_roc_auc"]); briers.extend(-sc["test_neg_brier_score"]); lls.extend(-sc["test_neg_log_loss"])
    aucs = np.asarray(aucs); briers = np.asarray(briers); lls = np.asarray(lls)
    print(f"\n=== {name} ({method}, K={n_components}, framing={framing}) ===", flush=True)
    print(f"ROC-AUC:  {aucs.mean():.3f} (SD {aucs.std(ddof=1):.3f}) "
          f"[{np.percentile(aucs,2.5):.3f}-{np.percentile(aucs,97.5):.3f}]", flush=True)
    print(f"Brier:    {briers.mean():.3f}", flush=True)
    print(f"Log-loss: {lls.mean():.3f}", flush=True)
    return {"auc": aucs}


def load_data():
    RAW_FILE = "adni_all_raw.raw"
    header = pd.read_csv(RAW_FILE, sep=r"\s+", nrows=0).columns.tolist()
    variant_cols = header[6:]
    raw = pd.read_csv(RAW_FILE, sep=r"\s+", dtype={c: np.float32 for c in variant_cols})
    raw["IID"] = raw["IID"].astype(str).str.replace("_s_", "_S_", regex=False).str.upper().str.strip()
    variant_cols = list(raw.columns[6:])
    geno = raw[["IID"] + variant_cols].rename(columns={"IID": "PTID"})
    COV_COLS = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4_COUNT", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]
    have_cov = (cov_data is not None) and all(c in cov_data.columns for c in COV_COLS)
    src = cov_data[["PTID", "ever_AD"] + COV_COLS] if have_cov else benchmark_data[["PTID", "ever_AD"]]
    src = src.copy(); src["PTID"] = src["PTID"].astype(str).str.upper().str.strip()
    merged = src.merge(geno, on="PTID", how="inner")
    print(f"Merged: n={len(merged)} (cases={int(merged['ever_AD'].sum())}, "
          f"controls={int((merged['ever_AD']==0).sum())}) covariates={have_cov}", flush=True)
    yg = merged["ever_AD"].to_numpy(np.int64)
    Xg = merged[variant_cols].to_numpy(np.float32)
    Xcov = merged[COV_COLS].to_numpy(np.float32) if have_cov else None
    return Xg, yg, Xcov, have_cov

if __name__ == "__main__":
    Xg, yg, Xcov, have_cov = load_data()
    print("\nPLS: Framing 2 (alone)", flush=True)
    run_supervised_benchmark(Xg, yg, method="pls", n_components=10, framing="alone", num_trials=30, name="PLS alone")
    if have_cov:
        print("\nPLS: Framing 1 (+covariates+APOE)", flush=True)
        Xcomb = np.hstack([Xg, Xcov]); n_geno = Xg.shape[1]
        run_supervised_benchmark(Xcomb, yg, method="pls", n_components=10, framing="covariates",
                                 n_geno=n_geno, num_trials=30, name="PLS +cov")
