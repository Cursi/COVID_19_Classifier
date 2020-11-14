import pandas as pd
import sys
from sklearn.utils import resample

df = pd.read_excel(sys.argv[1])
df_downsampled = resample(df, replace=False, n_samples=500, random_state=123)
df_downsampled.to_excel("generatedSubset.xlsx", index=False)