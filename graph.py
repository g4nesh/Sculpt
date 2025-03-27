import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.interpolate import UnivariateSpline

# Simulated data
np.random.seed(42)
ages = np.sort(np.random.lognormal(mean=1, sigma=0.5, size=100))
protein_levels = np.random.normal(loc=0, scale=1, size=100)
rna_levels = np.random.normal(loc=0.2, scale=1, size=100)

# Create DataFrame
df = pd.DataFrame({"Age": ages, "Protein": protein_levels, "RNA": rna_levels})

# Log scale transformation
df["Log_Age"] = np.log(df["Age"])

# Plot
plt.figure(figsize=(10, 6))
sns.scatterplot(x="Log_Age", y="Protein", data=df, label="Protein", color="blue", marker="D")
sns.scatterplot(x="Log_Age", y="RNA", data=df, label="RNA", color="red", marker="^")

# Fit smooth curves
sns.regplot(x="Log_Age", y="Protein", data=df, scatter=False, color="blue", order=3)
sns.regplot(x="Log_Age", y="RNA", data=df, scatter=False, color="red", order=3)

plt.xlabel("Age (log scale)")
plt.ylabel("Z-scores")
plt.title("Sculpt's Predicted Model for BCAN")
plt.legend()
plt.show()
