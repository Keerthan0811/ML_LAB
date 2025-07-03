# Build a KNN model for predicting if a person will have diabetes or not with a high accuracy
# score. Perform some appropriate Pre-Processing steps on the given dataset for better results.
# Implement the KNN algorithm on your own. Try other possible processes that can be done to
# dataset and tuning the model to increase accuracy such as Increase K value, Normalization and
# Different Distance Metrics. 



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Load diabetes dataset
data = load_diabetes()
X, y = data.data, (data.target > 140).astype(int)  # Binary: 1 = diabetic, 0 = non-diabetic

# Normalize the features
X = MinMaxScaler().fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=0)

# Store results
results = []

# Try different values of k and distance metrics
for k in [3, 5, 7,9]:
    for metric in ['euclidean', 'manhattan']:
        model = KNeighborsClassifier(n_neighbors=k, metric=metric)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        results.append({'K': k, 'Metric': metric, 'Accuracy': acc})
        print(f"K={k}, Metric={metric}, Accuracy={acc:.4f}")

# Convert results to DataFrame
df_result = pd.DataFrame(results)

# Plot Accuracy vs K
for metric in ['euclidean', 'manhattan']:
    subset = df_result[df_result['Metric'] == metric]
    plt.plot(subset['K'], subset['Accuracy'], marker='o', label=metric)

plt.title("KNN Accuracy vs K (Euclidean vs Manhattan)")
plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.xticks([3,5, 7,9])
plt.grid(True)
plt.legend()
plt.show()
