import pandas as pd

# reload geno_pca exactly as the benchmark does
geno_pca = pd.read_csv("geno_pca.eigenvec", sep=r"\s+")
print("raw columns:", list(geno_pca.columns)[:5], "...")
print("shape:", geno_pca.shape)
print("first IID value:", geno_pca.iloc[0, 1])