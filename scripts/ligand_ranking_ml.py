import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load data
data = pd.read_csv("../data/example_ligand_data.csv")

# Features and target
X = data[["mw", "logp", "hbond_donors", "hbond_acceptors"]]
y = data["binding_energy"]

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(
    n_estimators=100, random_state=42
)
model.fit(X_train, y_train)

# Predict binding energies
data["predicted_energy"] = model.predict(X)

# Rank ligands
data["rank"] = data["predicted_energy"].rank(ascending=True)

# Save ranked ligands
data_sorted = data.sort_values("rank")
print(data_sorted[["ligand_id", "predicted_energy", "rank"]])

# Feature importance plot
importances = model.feature_importances_
features = X.columns

plt.bar(features, importances)
plt.title("Feature Importance for Ligand Ranking")
plt.ylabel("Importance")
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig("../results/feature_importance.png")
plt.show()
