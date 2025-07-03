# Supervised Learning Algorithms - Logistic Regression (Univariant): Implement logistic
# regression and test it using any dataset. Give new test data and predict the classification
# output. Print the confusion matrix, accuracy, precision, recall, MSE , RMSE etc. Analyze
# and write the inference.

#5
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    r2_score, confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, mean_squared_error
)
from sklearn.preprocessing import StandardScaler


data = load_breast_cancer()
X= data.data
y = data.target
feature_names = data.feature_names
target_names = data.target_names

# Use only one feature ('mean radius' -> index 0)
x = X[:, 0].reshape(-1, 1)  # 'mean radius'


df = pd.DataFrame(X, columns=feature_names)
print("Missing values:", df.isnull().sum().sum())
print("Target classes:", np.unique(y))


scaler = StandardScaler()
x = scaler.fit_transform(x)


x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)


model = LogisticRegression()
model.fit(x_train, y_train)


y_pred = model.predict(x_test)


cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=target_names, yticklabels=target_names)
plt.title("Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.show()


print(f"\nr2_score: {r2_score(y_test, y_pred):.4f}")
print(f"accuracy_score: {accuracy_score(y_test, y_pred):.4f}")
print(f"precision_score: {precision_score(y_test, y_pred):.4f}")
print(f"recall_score: {recall_score(y_test, y_pred):.4f}")
print(f"mean_squared_error: {mean_squared_error(y_test, y_pred):.4f}")


x_plot = np.linspace(x.min(), x.max(), 300).reshape(-1, 1)
y_plot = model.predict_proba(x_plot)[:, 1]

plt.figure(figsize=(6, 4))
plt.plot(x_plot, y_plot, label="Logistic Curve", color="blue")
plt.scatter(x, y, color="red", label="Original Data", alpha=0.6)
plt.xlabel("Mean Radius")
plt.ylabel("Predicted Probability of Benign")
plt.title("Logistic Regression Curve (Breast Cancer Data)")
plt.legend()
plt.grid(True)
plt.show()

