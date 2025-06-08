# Build a KNN model for predicting if a person will have diabetes or not with a high accuracy
# score. Perform some appropriate Pre-Processing steps on the given dataset for better results.
# Implement the KNN algorithm on your own. Try other possible processes that can be done to
# dataset and tuning the model to increase accuracy such as Increase K value, Normalization and
# Different Distance Metrics. 

import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Load dataset from sklearn
data = fetch_openml(name="diabetes", version=1, as_frame=True)
df = data.frame


# Features and label
X = df.drop('class', axis=1)
y = df['class'].map({'tested_negative': 0, 'tested_positive': 1})

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Try different K values and metrics
for k in [3, 5, 7]:
    for metric in ['euclidean', 'manhattan']:
        model = KNeighborsClassifier(n_neighbors=k, metric=metric)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"K={k}, Metric={metric}, Accuracy={acc:.4f}")
