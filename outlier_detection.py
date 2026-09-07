import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")            
import matplotlib.pyplot as plt

PCS_FILE = "ADNI_WGS-TRANS.profiles"

PGS_FILES = {
    "alzheimers_pgs": "ADNI_WGS-GCST90704648-TRANS.profiles",
    "cad_pgs": "ADNI_WGS-GCST90132314-TRANS.profiles",
    "depression_pgs": "ADNI_WGS-GCST90726344-TRANS.profiles",
    "hypertension_pgs": "ADNI_WGS-GCST90475922-TRANS.profiles",
    "t2d_pgs": "ADNI_WGS-GCST90475667-TRANS.profiles"
}

#threshold for z-score outlier detection
Z_THRESH = 4.0 

#load the data
def load_pgs_data(csv_file, pgs_col=None):
    df = pd.read_csv(csv_file, sep=r"\s+")
    df["IID"] = df["IID"].str.replace("_s_", "_S_", regex=False)
    df["IID"] = df["IID"].astype(str).str.upper().str.strip()
    if pgs_col is not None:
        df = df.rename(columns={df.columns[2]: pgs_col})[["IID", pgs_col]]
    return df

#load pcs
pcs = load_pgs_data(PCS_FILE)
pc_cols = [c for c in pcs.columns if c.startswith("PC")]   # PC1..PC6
pcs = pcs[["IID"] + pc_cols].copy()
print(f"PCs loaded: {len(pcs)} people, {len(pc_cols)} PCs — {pc_cols}")

#merge all the data
merged = pcs.copy()
for name, path in PGS_FILES.items():
    s = load_pgs_data(path, pgs_col=name)
    merged = merged.merge(s, on="IID", how="inner")
print(f"Merged PGS+PCs: {len(merged)} people\n")


# find outliers on the two problem scores: hypertension and t2d
problem = ["hypertension_pgs", "t2d_pgs"]
merged["is_outlier"] = False
for c in problem:
    z = (merged[c] - merged[c].mean()) / merged[c].std()
    merged[c + "_z"] = z
    flag = z.abs() > Z_THRESH
    merged["is_outlier"] |= flag
    print(f"{c}: {flag.sum()} people with |z| > {Z_THRESH} "
          f"(max |z| = {z.abs().max():.1f})")
 
outliers = merged[merged["is_outlier"]].copy()
print(f"\nTotal distinct outlier individuals: {len(outliers)}")
print("\n--- Outlier individuals (IID, PC1, PC2, and z-scores) ---")
show_cols = ["IID", "PC1", "PC2"] + [c + "_z" for c in problem]
print(outliers[show_cols].round(2).to_string(index=False))
 
# save the IIDs
outliers[["IID"]].to_csv("outlier_iids.txt", index=False, header=False)
 
# plot PC1 vs PC2, highlight outliers
fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
 
for ax, (pcx, pcy) in zip(axes, [("PC1", "PC2"), ("PC1", "PC3")]):
    if pcy not in merged.columns:
        pcy = "PC2"
    normal = merged[~merged["is_outlier"]]
    ax.scatter(normal[pcx], normal[pcy], s=14, c="#B7C4DE",
               alpha=0.6, label="All participants", edgecolors="none")
    ax.scatter(outliers[pcx], outliers[pcy], s=90, c="#C0392B",
               marker="D", label=f"PGS outliers (|z|>{Z_THRESH})",
               edgecolors="black", linewidths=0.6, zorder=5)
    ax.set_xlabel(pcx); ax.set_ylabel(pcy)
    ax.set_title(f"{pcx} vs {pcy}")
    ax.legend(frameon=False, fontsize=9)
    ax.spines[["top", "right"]].set_visible(False)
 

fig.tight_layout(rect=[0, 0, 1, 0.93])
fig.savefig("outlier_pc_plot.png", dpi=150)
 
# print how far outliers are from the PC centre
centre = merged[pc_cols].mean()
spread = merged[pc_cols].std()
merged["pc_dist"] = np.sqrt(
    (((merged[pc_cols] - centre) / spread) ** 2).sum(axis=1)
)

outliers = merged[merged["is_outlier"]].copy()
print(f"\nTotal distinct outlier individuals: {len(outliers)}")
print("\n--- Outlier individuals (IID, PC1, PC2, z-scores, PC-distance) ---")
show_cols = ["IID", "PC1", "PC2"] + [c + "_z" for c in problem] + ["pc_dist"]
print(outliers[show_cols].round(2).to_string(index=False))

print(f"\n  All participants: median {merged['pc_dist'].median():.2f}, "
      f"95th pct {merged['pc_dist'].quantile(0.95):.2f}")
out_dist = merged.loc[merged["is_outlier"], "pc_dist"]
print(f"  Outliers median: {out_dist.median():.2f}")