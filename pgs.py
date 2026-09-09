#positive control for height/bmi pgs
import pandas as pd
import numpy as np
from scipy import stats
import pymc as pm
import pytensor.tensor as pt
import arviz as az
import statsmodels.formula.api as smf
import statsmodels.api as sm
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_val_score, cross_validate
from sklearn.linear_model import ElasticNet, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.exceptions import ConvergenceWarning
import matplotlib.pyplot as plt
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.metrics import roc_auc_score, roc_curve, auc, brier_score_loss, log_loss
from sklearn.ensemble import HistGradientBoostingClassifier
import warnings
warnings.filterwarnings("ignore") 
np.seterr(over="ignore", invalid="ignore", divide="ignore")
warnings.filterwarnings("ignore", message="overflow encountered")
warnings.filterwarnings("ignore", message="invalid value encountered")
warnings.filterwarnings("ignore", message="divide by zero encountered")

VITALS_CSV   = "VITALS.csv"
PTDEMOG_CSV  = "PTDEMOG.csv"
APOERES_CSV = "APOERES.csv"
HEIGHT_PGS   = "ADNI_WGS-GCST90565843-TRANS.profiles"   
BMI_PGS      = "ADNI_WGS-GCST90475153-TRANS.profiles" 
ALZHEIMERS_PGS = "ADNI_WGS-GCST90704648-TRANS.profiles"
CAD_PGS = "ADNI_WGS-GCST90132314-TRANS.profiles"
DEPRESSION_PGS = "ADNI_WGS-GCST90726344-TRANS.profiles"
HYPERTENSION_PGS = "ADNI_WGS-GCST90475922-TRANS.profiles"
T2D_PGS = "ADNI_WGS-GCST90475667-TRANS.profiles"
INTELLIGENCE_PGS = "ADNI_WGS-GCST006250-TRANS.profiles"
EDUCATION_PGS = "ADNI_WGS-GCST003676-TRANS.profiles"
PCS_FILE = "ADNI_WGS-TRANS.profiles"
DXSUM_CSV = "DXSUM.csv"

#create a function to load PGS data
def load_pgs_data(csv_file, pgs_col=None):
    df = pd.read_csv(csv_file, sep=r"\s+")
    df["IID"] = df["IID"].str.replace("_s_", "_S_", regex=False)
    df["IID"] = df["IID"].astype(str).str.upper().str.strip()
    if pgs_col is not None:
        df = df.rename(columns={df.columns[2]: pgs_col})[["IID", pgs_col]]
    return df

height_pgs = load_pgs_data(HEIGHT_PGS, "height_pgs")
bmi_pgs = load_pgs_data(BMI_PGS, "bmi_pgs")
alzheimers_pgs = load_pgs_data(ALZHEIMERS_PGS, "alzheimers_pgs")
cad_pgs = load_pgs_data(CAD_PGS, "cad_pgs")
depression_pgs = load_pgs_data(DEPRESSION_PGS, "depression_pgs")
hypertension_pgs = load_pgs_data(HYPERTENSION_PGS, "hypertension_pgs")
t2d_pgs = load_pgs_data(T2D_PGS, "t2d_pgs")
intelligence_pgs = load_pgs_data(INTELLIGENCE_PGS, "intelligence_pgs")
education_pgs = load_pgs_data(EDUCATION_PGS, "education_pgs")
pcs = load_pgs_data(PCS_FILE)
dxsum = pd.read_csv(DXSUM_CSV,  low_memory=False)
apoeres = pd.read_csv(APOERES_CSV,  low_memory=False)

# Load Genetic PCs 
pc_cols = [c for c in pcs.columns if c.startswith("PC")]   # PC1..PC6
pcs = pcs[["IID"] + pc_cols].copy()
print(f"PCs loaded: {len(pcs)} people, {len(pc_cols)} PCs — {pc_cols}")

vitals = pd.read_csv(VITALS_CSV, low_memory=False)
print(f"\nVITALS raw rows: {len(vitals)}")

vitals.columns = [c.upper() for c in vitals.columns]
 
# Keep only the columns needed
keep_columns = ["PTID", "VSHEIGHT", "VSHTUNIT", "VSWEIGHT", "VSWTUNIT"]
visit = "VISCODE2" if "VISCODE2" in vitals.columns else "VISCODE"
keep = [c for c in keep_columns if c in vitals.columns] + [visit]
vitals = vitals[keep].copy()

#keep the rows with values from the baseline visit (bl) and screening visit (sc), and initial visit (init)
vitals_bl_sc = vitals[vitals[visit].isin(["bl", "sc", "init"])].copy()
vitals_bl_sc = vitals_bl_sc[(vitals_bl_sc["VSHEIGHT"] > 0) & (vitals_bl_sc["VSWEIGHT"] > 0)].copy()
have_both = vitals_bl_sc.dropna(subset=["VSHEIGHT", "VSWEIGHT"]).copy()
have_both["_is_bl"] = have_both[visit].eq("sc") #prioritize screening
have_both = have_both.sort_values(["PTID", "_is_bl"], ascending=[True, False])
vitals_one = have_both.drop_duplicates(subset="PTID", keep="first").copy()
print(f"Total VITALS with height and weight: {len(vitals_one)}")


def inches_to_cm(df):
    if df["VSHTUNIT"] == 1:
        return df["VSHEIGHT"] * 2.54 
    else:
        return df["VSHEIGHT"]

def lbs_to_kg(df):
    if df["VSWTUNIT"] == 1:
        return df["VSWEIGHT"] * 0.453592
    else:
        return df["VSWEIGHT"]

vitals_one["PTID"] = vitals_one["PTID"].astype(str).str.upper().str.strip()
vitals_one["VSHEIGHT_CM"] = vitals_one.apply(inches_to_cm, axis=1)
vitals_one["VSWEIGHT_KG"] = vitals_one.apply(lbs_to_kg, axis=1)
vitals_one["BMI"] = vitals_one["VSWEIGHT_KG"] / (vitals_one["VSHEIGHT_CM"] / 100) ** 2

# Filter out implausible values for height, weight, and BMI
vitals_one = vitals_one[
    vitals_one["VSHEIGHT_CM"].between(140, 210) &
    vitals_one["VSWEIGHT_KG"].between(35, 200) &
    vitals_one["BMI"].between(14, 60)
].copy()

print("Measured height_cm:", vitals_one['VSHEIGHT_CM'].describe()[['mean','min','max']].to_dict())
print("Measured weight_kg:", vitals_one['VSWEIGHT_KG'].describe()[['mean','min','max']].to_dict())
print("Measured BMI:", vitals_one['BMI'].describe()[['mean','min','max']].to_dict())

#get age and sex from patient demographics for covariate adjustment
pt_demog = pd.read_csv(PTDEMOG_CSV, low_memory=False)
dkeep = [c for c in ["PTID", "PTGENDER", "PTDOBYY", "VISDATE", "PTEDUCAT"] if c in pt_demog.columns]
pt_demog = pt_demog[dkeep].drop_duplicates(subset="PTID", keep="first").copy()
pt_demog["PTID"] = pt_demog["PTID"].astype(str).str.upper().str.strip()
pt_demog["PTDOBYY"] = pd.to_datetime(pt_demog["PTDOBYY"], errors="coerce")
pt_demog["VISDATE"] = pd.to_datetime(pt_demog["VISDATE"], errors="coerce")
pt_demog["AGE"] = ((pt_demog["VISDATE"] - pt_demog["PTDOBYY"]).dt.days / 365.25).round().astype(int)


data = (vitals_one[["PTID", "VSHEIGHT_CM", "BMI"]]
        .merge(pt_demog, on="PTID", how="left")
        .merge(height_pgs.rename(columns={"IID": "PTID"}), on="PTID", how="inner")
        .merge(bmi_pgs.rename(columns={"IID": "PTID"}),    on="PTID", how="inner")
        .merge(pcs.rename(columns={"IID": "PTID"}),        on="PTID", how="inner"))
 
print(f"{len(data)} people with measured trait + PGS")

#positive control: check that height and BMI PGS correlate with measured height and BMI
def pos_control(col, pgs_col, df):
    df = df.copy()
    df[pgs_col + "_z"] = (df[pgs_col] - df[pgs_col].mean()) / df[pgs_col].std()
    df[col + "_z"] = (df[col] - df[col].mean()) / df[col].std()
    zcol = pgs_col + "_z"
    ycol = col + "_z"

    r, p = stats.pearsonr(df[zcol], df[col])
    r2 = r ** 2
    print(f"  Pearson r (PGS vs measured trait) = {r:.3f}   p = {p:.2e}")
    
    # variance explained by PGS alone
    print(f"  R^2 (PGS alone) = {r**2:.3f}  ({100*r**2:.1f}% of variance)")

    # model 1: regression adjusting for sex and age
    if ("PTGENDER" in df.columns and df["PTGENDER"].nunique() > 1) and "AGE" in df.columns:
        df["sex"] = df["PTGENDER"].astype("category")
        m1 = smf.ols(f"{ycol} ~ {zcol} + AGE + C(sex)", data=df).fit()
        print(f"  Adjusted for age + sex: standardised beta = {m1.params[zcol]:.3f}, "f"p = {m1.pvalues[zcol]:.2e}")
    
    #model 2: regression adjusting for sex, age, and genetic PCs
    pc_terms = " + ".join(pc_cols)
    m2 = smf.ols(f"{ycol} ~ {zcol} + AGE + C(sex) + {pc_terms}", data=df).fit()
    print(f"  Adjusted for age + sex + PCs: standardised beta = {m2.params[zcol]:.3f}, "f"p = {m2.pvalues[zcol]:.2e}")

    # incremental variance explained by the PGS
    m_null = smf.ols(f"{ycol} ~ AGE + C(sex) + {pc_terms}", data=df).fit()          # covariates only
    m_full = smf.ols(f"{ycol} ~ {zcol} + AGE + C(sex) + {pc_terms}", data=df).fit()  # + PGS
    delta_r2 = m_full.rsquared - m_null.rsquared
    print(f"  Model R^2: {100*m1.rsquared:.1f}% (no PCs) -> {100*m2.rsquared:.1f}% (with PCs)")
    print(f"  Incremental R^2 from PGS (Full - Null) = {100*delta_r2:.1f}%")

    return r, p
 
r_h, p_h = pos_control("VSHEIGHT_CM", "height_pgs", data)
r_b, p_b = pos_control("BMI", "bmi_pgs", data)


#DXSUM 
dxsum.columns = [c.upper() for c in dxsum.columns]
dxsum["PTID"] = dxsum["PTID"].astype(str).str.upper().str.strip()

# check for missing DIAGNOSIS
print("Total DXSUM rows:", len(dxsum))
dx = dxsum.dropna(subset=["DIAGNOSIS"]).copy()
print("Rows with valid DIAGNOSIS:", len(dx))
dx["EXAMDATE"] = pd.to_datetime(dx["EXAMDATE"], errors="coerce")
dx = dx.sort_values(["PTID", "EXAMDATE"])

# only keep the baseline visits
baseline_dx = (dx.drop_duplicates(subset="PTID", keep="first")
                 [["PTID", "DIAGNOSIS"]]
                 .rename(columns={"DIAGNOSIS": "dx_baseline"}))

# EVER-AD (did they ever receive a dementia diagnosis?)
# 3 = Dementia is the highest code (worst diagnosis)
ever_ad = (dx.groupby("PTID")["DIAGNOSIS"]
             .max()                     
             .rename("dx_worst")
             .reset_index())
ever_ad["ever_AD"] = (ever_ad["dx_worst"] == 3).astype(int)

# check how much follow-up each person has
followup = (dx.groupby("PTID")
              .agg(n_visits=("DIAGNOSIS", "size"),
                   first_visit=("EXAMDATE", "min"),
                   last_visit=("EXAMDATE", "max"))
              .reset_index())
followup["followup_years"] = (followup["last_visit"] - followup["first_visit"]).dt.days / 365.25

dx_labels = baseline_dx.merge(ever_ad, on="PTID").merge(followup, on="PTID")

print("\n--- BASELINE diagnosis (per person) ---")
print(dx_labels["dx_baseline"].value_counts().sort_index())
print("\n--- WORST-EVER diagnosis (per person) ---")
print(dx_labels["dx_worst"].value_counts().sort_index())
print("\nMedian follow-up (years):", dx_labels["followup_years"].median().round(1))
print("\nConverters (CN/MCI at baseline -> Dementia later):",
      ((dx_labels["dx_baseline"] < 3) & (dx_labels["dx_worst"] == 3)).sum())

# worst-ever diagnosis: 1=CN, 2=MCI, 3=Dementia
ever = dx.groupby("PTID")["DIAGNOSIS"].max().rename("dx_worst").reset_index()

#primary contrast: ever-AD (3) vs ever-CN (1); DROP ever-MCI (2)
ever["ever_AD"] = np.where(ever["dx_worst"] == 3, 1,
                   np.where(ever["dx_worst"] == 1, 0, np.nan))  # MCI -> NaN
dx_labels = ever.merge(followup, on="PTID")


ad_covariates = (pt_demog.merge(pcs.rename(columns={"IID": "PTID"}),on="PTID",how="inner"))

#Choose APOERES columns
apoeres = apoeres[["PTID", "GENOTYPE"]].copy()
apoeres["PTID"] = (apoeres["PTID"].astype(str).str.replace("_s_", "_S_", regex=False).str.upper().str.strip())
apoeres["GENOTYPE"] = apoeres["GENOTYPE"].astype(str).str.strip()
apoeres = apoeres.drop_duplicates(subset="PTID", keep="first")

# Count the number of APOE4 alleles (0, 1, or 2) for each individual
def count_e4(g):
    a, b = g.split("/")
    return (a == "4") + (b == "4")

apoeres["APOE4_COUNT"] = apoeres["GENOTYPE"].apply(count_e4)

# Merge all the data together for modeling
model_data = (dx_labels[["PTID", "ever_AD"]]
              .merge(ad_covariates, on="PTID", how="inner")
              .merge(apoeres, on="PTID", how="inner")
              .merge(alzheimers_pgs.rename(columns={"IID": "PTID"}), on="PTID", how="inner")
              .merge(depression_pgs.rename(columns={"IID": "PTID"}), on="PTID", how="inner")
              .merge(cad_pgs.rename(columns={"IID": "PTID"}), on="PTID", how="inner")
              .merge(hypertension_pgs.rename(columns={"IID": "PTID"}), on="PTID", how="inner")
              .merge(t2d_pgs.rename(columns={"IID": "PTID"}), on="PTID", how="inner")
              .merge(intelligence_pgs.rename(columns={"IID": "PTID"}), on="PTID", how="inner")
              .merge(education_pgs.rename(columns={"IID": "PTID"}), on="PTID", how="inner")
)
model_data = model_data.dropna(subset=["ever_AD"]).copy()
model_data["ever_AD"] = model_data["ever_AD"].astype(int)
model_data["PTGENDER"] = (model_data["PTGENDER"] == 2).astype(int)  # 1=female, 0=male (check coding)

'''
#EXCLUDE GENETIC-ANCESTRY OUTLIERS (PC-distance > 95th pct)

centre = model_data[pc_cols].mean()
spread = model_data[pc_cols].std()
model_data["pc_dist"] = np.sqrt(
    (((model_data[pc_cols] - centre) / spread) ** 2).sum(axis=1)
)
pc_thresh = model_data["pc_dist"].quantile(0.95)
ancestry_outliers = model_data["pc_dist"] > pc_thresh

print(f"\nPC-distance threshold (95th pct): {pc_thresh:.2f}")
print(f"Excluding {ancestry_outliers.sum()} ancestry outliers")
model_data = model_data[~ancestry_outliers].drop(columns="pc_dist").copy()
print(f"After exclusion: n={len(model_data)} "
      f"| cases={model_data['ever_AD'].sum()} "
      f"| controls={(model_data['ever_AD']==0).sum()}")
print(len(model_data))
'''
print(len(model_data))

# PENALIZED LOGISTIC REGRESSION MODEL WITH CROSS-VALIDATION
def run_logistic_regression(X, y, param_grid, num_trials=30, name=""):
    X_arr = np.asarray(X, dtype=np.float64)
    y_arr = np.asarray(y, dtype=np.int64)

    if np.isnan(X_arr).any() or np.isinf(X_arr).any():
        raise ValueError("Data matrix still contains hidden NaN or Inf values!")

    model = Pipeline([
    ("scaler", StandardScaler()),
    ("model", LogisticRegression(
        penalty="l2",
        solver="lbfgs",
        max_iter=5000,
        random_state=1
    ))
])
    pipeline_param_grid = {"model__C": param_grid["C"]}
    auc_scores, brier_scores, logloss_scores = [], [], []

    for i in range(num_trials):
        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=i)
        outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=i)

        # Nested CV with parameter optimization
        clf = GridSearchCV(estimator=model, param_grid=pipeline_param_grid, cv=inner_cv, scoring="roc_auc", n_jobs=1, error_score="raise")
        nested_score = cross_validate(clf, X=X, y=y, cv=outer_cv, scoring=["roc_auc", "neg_brier_score", "neg_log_loss"], n_jobs=1, error_score="raise")
        
        auc_scores.extend(nested_score["test_roc_auc"])
        brier_scores.extend(-nested_score["test_neg_brier_score"])
        logloss_scores.extend(-nested_score["test_neg_log_loss"])

    auc_scores = np.asarray(auc_scores)
    brier_scores = np.asarray(brier_scores)
    logloss_scores = np.asarray(logloss_scores)

    print(f"\n=== {name} ===")
    print(f"ROC-AUC:  {auc_scores.mean():.3f} (SD {auc_scores.std(ddof=1):.3f}) "
          f"[{np.percentile(auc_scores,2.5):.3f}–{np.percentile(auc_scores,97.5):.3f}]")
    print(f"Brier:    {brier_scores.mean():.3f} (SD {brier_scores.std(ddof=1):.3f})   (lower = better)")
    print(f"Log-loss: {logloss_scores.mean():.3f} (SD {logloss_scores.std(ddof=1):.3f})   (lower = better)")
    return {"auc": auc_scores, "brier": brier_scores, "logloss": logloss_scores}

cols = ["AGE","PTGENDER","PTEDUCAT","APOE4_COUNT","PC1","PC2","PC3","PC4","PC5","PC6",
        "alzheimers_pgs","depression_pgs","cad_pgs","hypertension_pgs","t2d_pgs","intelligence_pgs","education_pgs"]

# clip FIRST
pgs_cols = ["alzheimers_pgs","depression_pgs","cad_pgs","hypertension_pgs","t2d_pgs","intelligence_pgs","education_pgs"]
for c in pgs_cols:
    z = (model_data[c] - model_data[c].mean()) / model_data[c].std()
    model_data[c] = z.clip(-5, 5)

print("After clip, PGS ranges:")
print(model_data[pgs_cols].describe().T[["min","max"]])

X = model_data[cols]

covariates_column = ["AGE", "PTGENDER", "PTEDUCAT"]
covariates_column_pcs = ["AGE", "PTGENDER", "PTEDUCAT", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]
covariates_column_pcs_apoe = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4_COUNT", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]
ad_pgs_column = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4_COUNT", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "alzheimers_pgs"]
predictor_columns = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4_COUNT", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "alzheimers_pgs",
"depression_pgs", "cad_pgs", "hypertension_pgs", "t2d_pgs", "intelligence_pgs", "education_pgs"]
param_grid = {"C": [0.01, 0.03, 0.1, 0.3, 1, 3], "l1_ratio": [0.25, 0.5, 0.75]}

X_baseline_covariates = model_data[covariates_column].copy()
X_covariates_pcs = model_data[covariates_column_pcs].copy()
X_covariates_pcs_apoe = model_data[covariates_column_pcs_apoe].copy()
X_AD_pgs = model_data[ad_pgs_column].copy()
X_comorbidities_pgs = model_data[predictor_columns].copy()
X_AD_CAD_pgs = X_comorbidities_pgs.drop(columns=["depression_pgs", "hypertension_pgs", "t2d_pgs", "intelligence_pgs", "education_pgs"]).copy()
X_AD_depression_pgs = X_comorbidities_pgs.drop(columns=["hypertension_pgs", "t2d_pgs", "cad_pgs", "intelligence_pgs", "education_pgs"]).copy()
X_AD_hypertension_pgs = X_comorbidities_pgs.drop(columns=["depression_pgs", "t2d_pgs", "cad_pgs", "intelligence_pgs", "education_pgs"]).copy()
X_AD_t2d_pgs = X_comorbidities_pgs.drop(columns=["depression_pgs", "hypertension_pgs", "cad_pgs", "intelligence_pgs", "education_pgs"]).copy()
X_AD_intelligence_edu = X_comorbidities_pgs.drop(columns=["depression_pgs", "hypertension_pgs", "t2d_pgs", "cad_pgs"]).copy()
X_AD_intellince = model_data[ad_pgs_column + ["intelligence_pgs"]].copy()
X_AD_edu = model_data[ad_pgs_column + ["education_pgs"]].copy()
X_AD_covs = X_AD_pgs.drop(columns = ["APOE4_COUNT"]).copy() #Model 2 + AD-PGS without APOE4
y = model_data["ever_AD"].copy()

print("model_data:", len(model_data), "| cases:", model_data["ever_AD"].sum(),
      "| controls:", (model_data["ever_AD"]==0).sum())
run_logistic_regression(X_baseline_covariates, y, param_grid, num_trials=30, name="Baseline Covariates")
run_logistic_regression(X_covariates_pcs, y, param_grid, num_trials=30, name="Covariates with PCs")
run_logistic_regression(X_covariates_pcs_apoe, y, param_grid, num_trials=30, name="Covariates with PCs and APOE4")
run_logistic_regression(X_AD_pgs, y, param_grid, num_trials=30, name="AD PGS")
run_logistic_regression(X_AD_CAD_pgs, y, param_grid, num_trials=30, name="AD CAD PGS")
run_logistic_regression(X_AD_depression_pgs, y, param_grid, num_trials=30, name="AD Depression PGS")
run_logistic_regression(X_AD_hypertension_pgs, y, param_grid, num_trials=30, name="AD Hypertension PGS")
run_logistic_regression(X_AD_t2d_pgs, y, param_grid, num_trials=30, name="AD T2D PGS")
run_logistic_regression(X_AD_intelligence_edu, y, param_grid, num_trials=30, name="AD Intelligence and Education PGS")
run_logistic_regression(X_comorbidities_pgs, y, param_grid, num_trials=30, name="Comorbidities PGS")
run_logistic_regression(X_AD_intellince, y, param_grid, num_trials=30, name="AD Intelligence PGS")
run_logistic_regression(X_AD_edu, y, param_grid, num_trials=30, name="AD Education PGS")
run_logistic_regression(X_AD_covs, y, param_grid, num_trials=30, name="AD PGS without APOE4")

#gradient boosting model for comparison
def run_xgb_model(X, y, num_trials=30, name=""):
    X_arr = np.asarray(X, dtype=np.float64)
    if np.isnan(X_arr).any() or np.isinf(X_arr).any():
        raise ValueError("Data matrix contains NaN or Inf!")

    model = HistGradientBoostingClassifier(random_state=1)

    gb_param_grid = {
    "max_depth": [2, 3],
    "learning_rate": [0.05],
    "max_iter": [200],
    "min_samples_leaf": [20, 40],
    "l2_regularization": [1.0],
    }

    auc_scores, brier_scores, logloss_scores = [], [], []

    for i in range(num_trials):
        inner_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=i)
        outer_cv = StratifiedKFold(n_splits=4, shuffle=True, random_state=i)

        # Nested CV with parameter optimization
        clf = GridSearchCV(estimator=model, param_grid=gb_param_grid, cv=inner_cv, scoring="roc_auc", n_jobs=1, error_score="raise")
        nested_score = cross_validate(clf, X=X, y=y, cv=outer_cv, scoring=["roc_auc", "neg_brier_score", "neg_log_loss"], n_jobs=-1, error_score="raise")
        
        auc_scores.extend(nested_score["test_roc_auc"])
        brier_scores.extend(-nested_score["test_neg_brier_score"])
        logloss_scores.extend(-nested_score["test_neg_log_loss"])

    auc_scores = np.asarray(auc_scores)
    brier_scores = np.asarray(brier_scores)
    logloss_scores = np.asarray(logloss_scores)

    print(f"\n=== {name} ===")
    print(f"ROC-AUC:  {auc_scores.mean():.3f} (SD {auc_scores.std(ddof=1):.3f}) "
          f"[{np.percentile(auc_scores,2.5):.3f}–{np.percentile(auc_scores,97.5):.3f}]")
    print(f"Brier:    {brier_scores.mean():.3f} (SD {brier_scores.std(ddof=1):.3f})   (lower = better)")
    print(f"Log-loss: {logloss_scores.mean():.3f} (SD {logloss_scores.std(ddof=1):.3f})   (lower = better)")
    return {"auc": auc_scores, "brier": brier_scores, "logloss": logloss_scores}

run_xgb_model(X_AD_pgs, y, num_trials=30, name="AD PGS Gradient Boosting")
run_xgb_model(X_comorbidities_pgs, y, num_trials=30, name="Comorbidities PGS Gradient Boosting")
run_xgb_model(X_AD_intelligence_edu, y, num_trials=30, name="AD Intelligence and Education PGS Gradient Boosting")

#bootstrap optimism correction for ROC-AUC as a robustness check
def bootstrap_opt(X, y, num_bootsraps=500, name="") :
    X = X.reset_index(drop=True)
    y = y.reset_index(drop=True)
    X_arr = np.asarray(X, dtype=np.float64)
    if np.isnan(X_arr).any() or np.isinf(X_arr).any():
        raise ValueError("Data matrix contains NaN or Inf!")
    
    rng = np.random.default_rng(1)

    base = Pipeline([
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(penalty="l2", solver="lbfgs",
                                         max_iter=5000, random_state=1))
        ])

    boot_param_grid = {"model__C": [0.01,0.03,0.1,0.3,1,3]}
    cv=StratifiedKFold(4, shuffle=True, random_state=0)
    clf = GridSearchCV(estimator=base, param_grid=boot_param_grid, cv=cv, scoring="roc_auc", n_jobs=1)
    model = clf.fit(X, y)
    app_score = roc_auc_score(y, model.predict_proba(X)[:, 1])

    optimism_scores = []
    for _ in range(num_bootsraps):
        indices = rng.choice(len(X), size=len(X), replace=True)
        X_bootstrap = X.iloc[indices]
        y_bootstrap = y.iloc[indices]
        if y_bootstrap.nunique() < 2: # Skip bootstrap sample if it contains only one class
            continue

        bootstrap_clf =GridSearchCV(estimator=base, param_grid=boot_param_grid, cv=cv, scoring="roc_auc", n_jobs=1)
        model_bootstrap = bootstrap_clf.fit(X_bootstrap, y_bootstrap) #refit on bootstrap sample
        bootstrap_score = roc_auc_score(y_bootstrap, model_bootstrap.predict_proba(X_bootstrap)[:, 1])
        original_score = roc_auc_score(y, model_bootstrap.predict_proba(X)[:, 1])
        optimism_scores.append(bootstrap_score - original_score) 
    
    mean_optimism_score = np.mean(optimism_scores)
    corrected_score = app_score - mean_optimism_score
    print(f"\n=== {name} ===")
    print(f"Apparent ROC-AUC: {app_score:.3f}")
    print(f"Mean Optimism Score: {mean_optimism_score:.3f}")
    print(f"Corrected ROC-AUC: {corrected_score:.3f}")
    return corrected_score

bootstrap_opt(X_AD_pgs, y, num_bootsraps=500, name="AD PGS Bootstrap Optimisation")
bootstrap_opt(X_comorbidities_pgs, y, num_bootsraps=500, name="Comorbidities PGS Bootstrap Optimisation")
bootstrap_opt(X_AD_intelligence_edu, y, num_bootsraps=500, name="AD Intelligence and Education PGS Bootstrap Optimisation")

#likelihood ratio test for nested models
def likelihood_ratio_test(model1, model2):
    ll1 = model1.llf
    ll2 = model2.llf
    df1 = model1.df_model
    df2 = model2.df_model
    lr_stat = 2 * (ll2 - ll1)
    df_diff = df2 - df1
    p_value = stats.chi2.sf(lr_stat, df_diff)
    return lr_stat, df_diff, p_value

def fit_logit(feature_cols):
    Xf = model_data[feature_cols].astype(float).copy()
    Xf = pd.DataFrame(StandardScaler().fit_transform(Xf),
                      columns=feature_cols, index=model_data.index)
    Xf = sm.add_constant(Xf)
    return sm.Logit(model_data["ever_AD"].values, Xf).fit(disp=0)

como_only_cols = ad_pgs_column + ["depression_pgs", "cad_pgs", "hypertension_pgs", "t2d_pgs"]

m_covpc_apoe = fit_logit(covariates_column_pcs_apoe)      # M3
m_ad         = fit_logit(ad_pgs_column)                   # M4 = M3 + AD-PGS
m_ad_cog     = fit_logit(ad_pgs_column + ["intelligence_pgs", "education_pgs"])  # M4 + cognitive
m_ad_como    = fit_logit(como_only_cols)                # M4 + everything
intel_cols = fit_logit(ad_pgs_column + ["intelligence_pgs"]) # M4 + intelligence
edu_cols   = fit_logit(ad_pgs_column + ["education_pgs"]) # M4 + education

print("\n LIKELIHOOD-RATIO TESTS")

# Does the AD-PGS add over APOE + covariates?
lr, ddf, p = likelihood_ratio_test(m_covpc_apoe, m_ad)
print(f"M3 vs M4 (add AD-PGS):              LR={lr:.2f}, df={ddf}, p={p:.4f}")

# Do the cognitive scores add over M4?
lr, ddf, p = likelihood_ratio_test(m_ad, m_ad_cog)
print(f"M4 vs M4+cognitive (add intel+edu): LR={lr:.2f}, df={ddf}, p={p:.4f}")

# Do the comorbidity scores add over M4?
lr, ddf, p = likelihood_ratio_test(m_ad, m_ad_como)
print(f"M4 vs M4+all comorbidities:         LR={lr:.2f}, df={ddf}, p={p:.4f}")

# Do the intelligence score add over M4?
lr, ddf, p = likelihood_ratio_test(m_ad, intel_cols)
print(f"M4 vs M4+intelligence:              LR={lr:.2f}, df={ddf}, p={p:.4f}")

# Do the educational attainment score add over M4?
lr, ddf, p = likelihood_ratio_test(m_ad, edu_cols)
print(f"M4 vs M4+education:                  LR={lr:.2f}, df={ddf}, p={p:.4f}")


def standardize(var) :
    standardized = (model_data[var] - model_data[var].mean()) / model_data[var].std()
    return standardized

# Bayesian logistic regression with horseshoe prior for comorbidities
def bayesian_horseshoe_prior(p0=2): #p0 = Prior guess, 2 of the 6 uncertain PGS are not null
    with pm.Model() as model:
        hs_cols = ["depression_pgs","cad_pgs","hypertension_pgs","t2d_pgs","intelligence_pgs","education_pgs"]
        
        pgs_array = model_data[hs_cols].to_numpy()
        educ_array = standardize("PTEDUCAT").to_numpy()
        std_apoe_array = standardize("APOE4_COUNT").to_numpy()
        pcs_array = model_data[pc_cols].to_numpy()
        age_array = standardize("AGE").to_numpy()
        sex_array = model_data["PTGENDER"].to_numpy()
        ad_pgs_array = model_data["alzheimers_pgs"].to_numpy()
        y_array = model_data["ever_AD"].to_numpy()

        D  = pgs_array.shape[1]
        N  = len(model_data)

        X_pgs = pm.Data("X_pgs", pgs_array)
        X_pcs = pm.Data("X_pcs", pcs_array)
        age_data = pm.Data("age", age_array)
        sex_data = pm.Data("sex", sex_array)
        apoe_data = pm.Data("apoe4", std_apoe_array)
        educ_data = pm.Data("education", educ_array)
        AD_pgs = pm.Data("AD_pgs", ad_pgs_array)
        y_data = pm.Data("y", y_array)

        intercept = pm.Normal("intercept", mu=0, sigma=1.5)

        # Priors for fixed adjustment covariates
        beta_age = pm.Normal("beta_age", mu=0, sigma=1)
        beta_sex = pm.Normal("beta_sex", mu=0, sigma=1)
        beta_educ = pm.Normal("beta_educ", mu=0, sigma=1)
        beta_apoe = pm.Normal("beta_apoe", mu=0, sigma=1.5)
        beta_pcs = pm.Normal("beta_pcs", mu=0, sigma=0.5, shape=len(pc_cols))
        beta_ad_pgs = pm.Normal("beta_ad_pgs", mu=0, sigma=1.5)
        
        
        sigma = 2.0
        tau0 = (p0 / (D - p0)) * (sigma / np.sqrt(N))
        print(f"\n[p0={p0}]  tau0 = {tau0:.4f}")

        tau = pm.HalfStudentT("tau", nu=1, sigma=tau0)
        lam = pm.HalfStudentT("lam", nu=1, sigma=1, shape=D)
        c2 = pm.InverseGamma("c2", alpha=2, beta=8)
        lam_tilde = pm.Deterministic("lam_tilde", pt.sqrt(c2 * lam**2 / (c2 + tau**2 * lam**2)))
        z = pm.Normal("z_pgs", mu=0, sigma=1, shape=D)
        # Put the horseshoe on pgs
        beta_pgs = pm.Deterministic("beta_pgs", z * tau * lam_tilde)

        # Linear predictor
        eta = (intercept + beta_age * age_data + beta_sex * sex_data + beta_apoe * apoe_data + beta_educ * educ_data 
               + beta_ad_pgs * AD_pgs + pm.math.dot(X_pcs, beta_pcs) + pm.math.dot(X_pgs, beta_pgs))
        p = pm.math.sigmoid(eta)
        
        # Binary AD outcome
        outcome = pm.Bernoulli("outcome", p=p, observed=y_data)
        trace = pm.sample(draws=2000, tune=2000, chains=4, target_accept=0.99)
        print("divergences:", trace.sample_stats["diverging"].sum().item())
        print(az.summary(trace, var_names=["beta_pgs","beta_ad_pgs","beta_apoe","tau"], hdi_prob=0.95))
        # Forest plot
        axes = az.plot_forest(trace, var_names=["beta_pgs", "beta_ad_pgs", "beta_apoe"], combined=True, figsize=(10, 5.5), colors="#1d3557", hdi_prob=0.94, quartiles=True)
        ax = axes[0]

        # Reference line for no effect
        ax.axvline(0, linestyle="--", linewidth=1, alpha=0.7)
        # Cleaner labels
        label_map = {
        "beta_pgs[0]": "Depression PGS",
        "[1]": "CAD PGS",
        "[2]": "Hypertension PGS",
        "[3]": "T2D PGS",
        "[4]": "Intelligence PGS",
        "[5]": "Education PGS",
        "beta_ad_pgs": "AD PGS",
        "beta_apoe": "APOE4 count"
        }

        current_labels = [tick.get_text() for tick in ax.get_yticklabels()]
        clean_labels = [label_map.get(label, label) for label in current_labels]

        ax.set_yticks(ax.get_yticks())
        ax.set_yticklabels(clean_labels)

        # Titles and axis labels
        ax.set_title("Posterior Coefficient Estimates", fontsize=15, pad=14)
        ax.set_xlabel("Posterior coefficient", fontsize=12)

        ax.set_ylabel("")

        # Improve readability
        ax.tick_params(axis="both", labelsize=11)
        ax.grid(axis='x', linestyle=':', alpha=0.5, color='#BBBBBB')

        # Remove unnecessary borders
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#CCCCCC")                       
        ax.spines["bottom"].set_color("#CCCCCC")   

        plt.tight_layout()
        plt.show()
    return model, trace

# sensitivity check for the prior
for p0_val in [1, 2, 3]:
    print(f"\n{'='*60}\nHORSESHOE SENSITIVITY: p0 = {p0_val}\n{'='*60}")
    #bayesian_horseshoe_prior(p0=p0_val)

# BENCHMARK: PGS vs Genotype-PCA representation
GENO_PCA_FILE = "geno_pca.eigenvec"

geno_pca = pd.read_csv(GENO_PCA_FILE, sep=r"\s+")

# drop FID (col 0), keep IID + 100 PCs
geno_pca = geno_pca.iloc[:, 1:]                      
geno_pca.columns = ["IID"] + [f"gPC{i}" for i in range(1, geno_pca.shape[1])]
geno_pca["IID"] = (geno_pca["IID"].astype(str)
                   .str.replace("_s_", "_S_", regex=False)
                   .str.upper().str.strip())

# merge
benchmark_data = model_data.merge(geno_pca.rename(columns={"IID": "PTID"}), on="PTID", how="inner")
print(f"\nBenchmark merged n={len(benchmark_data)} (model_data n={len(model_data)})")
print("  has alzimers_pgs?", "alzimers_pgs" in benchmark_data.columns)
print("  has gPC1?", "gPC1" in benchmark_data.columns)

benchmark_y = benchmark_data["ever_AD"].copy()

# shared columns: covariates + APOE4
shared_cols = ["AGE", "PTGENDER", "PTEDUCAT", "APOE4_COUNT"]

# Run the genotype-PCA arm for several K
print("\nBENCHMARK: genotype-PCA representation")
for K in [1, 2, 5, 10, 20, 50, 100]:
    gpc_cols = [f"gPC{i}" for i in range(1, K + 1)]
    X_geno = benchmark_data[shared_cols + gpc_cols].copy()
    run_logistic_regression(X_geno, benchmark_y, param_grid, num_trials=30, name=f"Genotype-PCA  (K={K})  + covariates + APOE")
    print(len(X_geno. columns))

# The PGS arm for direct comparison (shared covariates + APOE + AD-PGS)
print("\nBENCHMARK: PGS representation")
X_pgs_bench = benchmark_data[shared_cols + ["PC1","PC2","PC3","PC4","PC5","PC6","alzheimers_pgs"]].copy()
run_logistic_regression(X_pgs_bench, benchmark_y, param_grid, num_trials=30, name="PGS (AD-PGS)  + covariates + APOE + ancestry PCs")


print("\nBENCHMARK: representations alone, no covariates")
X_pgs_bench_ad_only = benchmark_data[["alzheimers_pgs"]].copy()
run_logistic_regression(X_pgs_bench_ad_only, benchmark_y, param_grid, num_trials=30, name="AD-PGS alone (no covariates)")

# genotype-PCs alone, for each K
for K in [1, 2, 5, 10, 20, 50, 100]:
    gpc_cols = [f"gPC{i}" for i in range(1, K + 1)]
    X_geno_only = benchmark_data[gpc_cols].copy()
    run_logistic_regression(X_geno_only, benchmark_y, param_grid, num_trials=30, name=f"Genotype-PCA alone (K={K}, no covariates)")

benchmark_data[["PTID", "ever_AD"]].to_csv("bench_labels.csv", index=False)
benchmark_data[["PTID", "ever_AD", "AGE", "PTGENDER", "PTEDUCAT", "APOE4_COUNT", "PC1", "PC2", "PC3", "PC4", "PC5", "PC6"]].to_csv("benchmark_data_with_covariates.csv", index=False)