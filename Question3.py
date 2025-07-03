# Design an experiment to investigate the impact of varying the number of trees in a Random
# Forest classifier on its performance for a given dataset. Write Python code to implement the
# Random Forest algorithm with different numbers of trees and evaluate its classification
# performance using appropriate evaluation metrics.


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# 1) Choose dataset: Breast Cancer Wisconsin (binary, 30 features, n=569)
data = load_breast_cancer()
X, y = data.data, data.target
feature_names = data.feature_names
# 2) Preprocessing: scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3) Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.3, random_state=42, stratify=y
)

# 4) Experiment: vary n_estimators
tree_counts = [1, 5, 10, 20, 50, 100, 200]
metrics = {'accuracy': [], 'precision': [], 'recall': [], 'f1': []}

for n in tree_counts:
    clf = RandomForestClassifier(n_estimators=n, criterion='entropy', random_state=42)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    metrics['accuracy'].append(accuracy_score(y_test, y_pred))
    metrics['precision'].append(precision_score(y_test, y_pred))
    metrics['recall'].append(recall_score(y_test, y_pred))
    metrics['f1'].append(f1_score(y_test, y_pred))
print("Performance Metrics:")
for i, n in enumerate(tree_counts):
    print(f"\nNumber of Trees: {n}")
    print(f"Accuracy: {metrics['accuracy'][i]:.4f}")
    print(f"Precision: {metrics['precision'][i]:.4f}")
    print(f"Recall: {metrics['recall'][i]:.4f}")
    print(f"F1-Score: {metrics['f1'][i]:.4f}")
# 5) Plotting
plt.figure(figsize=(10, 6))
for metric, marker in zip(metrics, ['o', 's', '^', 'x']):
    plt.plot(tree_counts, metrics[metric], marker=marker, label=metric.title())
plt.xlabel('Number of Trees (n_estimators)')
plt.ylabel('Score')
plt.title('Random Forest Performance vs. Number of Trees\n(Breast Cancer Wisconsin)')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# 6) Interpretation:
# Increasing the number of trees typically improves stability and generalization,
# with diminishing returns beyond ~50–100 trees.

