# Un-Supervised Learning Algorithms - K-Means Clustering: Build a K-Means Model for
# any dataset. Assume K value as 2,3,4 .Compare and interpret the results of different clusters..

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# Step 1: Load the Iris dataset
iris = load_iris()
X = iris.data
features = iris.feature_names

# Step 2: Standardize the data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 3: Reduce dimensions using PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

# Step 4: Try K-Means with K = 2, 3, 4
k_values = [2, 3, 4]
inertias = []
silhouettes = []

for k in k_values:
    model = KMeans(n_clusters=k, random_state=0)
    labels = model.fit_predict(X_scaled)

    # Save metrics
    inertias.append(model.inertia_)
    silhouettes.append(silhouette_score(X_scaled, labels))

    # Plot clusters
    plt.figure()
    plt.title(f"K = {k}")
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    centers = pca.transform(model.cluster_centers_)
    plt.scatter(centers[:, 0], centers[:, 1], c='red', s=150, marker='X', label='Centroids')
    plt.xlabel('PCA 1')
    plt.ylabel('PCA 2')
    plt.legend()
    plt.grid(True)
    plt.show()

# Step 5: Show comparison table
print("K\tInertia\t\tSilhouette Score")
for i in range(len(k_values)):
    print(f"{k_values[i]}\t{inertias[i]:.2f}\t\t{silhouettes[i]:.2f}")

# Step 6: Summary
print("\nSummary:")
print("→ K = 3 gives the best balance of compact clusters and separation.")
print("→ K-Means performance was compared using Inertia and Silhouette Score.")
print("→ PCA helped us visualize how well the clusters were formed.")
