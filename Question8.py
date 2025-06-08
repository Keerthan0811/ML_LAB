# Un-Supervised Learning Algorithms - K-Means Clustering: Build a K-Means Model for
# any dataset. Assume K value as 2,3,4 .Compare and interpret the results of different clusters..

from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Load data
data = load_iris()
X = data.data

# Reduce to 2D for visualization
X_pca = PCA(n_components=2).fit_transform(X)

# Try K = 2, 3, 4
ks = [2, 3, 4]
plt.figure(figsize=(12, 3))

for i, k in enumerate(ks, 1):
    km = KMeans(n_clusters=k, random_state=0)
    labels = km.fit_predict(X)
    plt.subplot(1, 3, i)
    plt.scatter(X_pca[:, 0], X_pca[:, 1], c=labels)
    plt.title(f"K={k}")
    plt.xlabel("PC1"); plt.ylabel("PC2")

plt.suptitle("K-Means Clustering with Different K Values")
plt.tight_layout(); plt.show()
