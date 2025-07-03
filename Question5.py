# Supervised Learning Algorithms - Logistic Regression (Univariant): Implement logistic
# regression and test it using any dataset. Give new test data and predict the classification
# output. Print the confusion matrix, accuracy, precision, recall, MSE , RMSE etc. Analyze
# and write the inference.



import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, accuracy_score, precision_score, recall_score, mean_squared_error
import matplotlib.pyplot as plt

# Data
X = np.array([[30], [40], [45], [49], [50], [55], [60], [65], [70], [80]])
y = np.array([0, 0, 0, 0, 1, 1, 1, 1, 1, 1])

# Train model
model = LogisticRegression().fit(X, y)

# Predict on training data
y_pred = model.predict(X)

# Metrics
print("Confusion Matrix:\n", confusion_matrix(y, y_pred))
print("Accuracy:", accuracy_score(y, y_pred))
print("Precision:", precision_score(y, y_pred))
print("Recall:", recall_score(y, y_pred))
mse = mean_squared_error(y, y_pred) # Calculate MSE first
print("MSE:", mse)
rmse = np.sqrt(mse) # Calculate RMSE manually
print("RMSE:", rmse)

# Predict on new data
print("New Predictions for [47], [53], [67]:", model.predict([[47], [53], [67]]))

# Plot
x_vals = np.linspace(30, 80, 200).reshape(-1, 1)
probs = model.predict_proba(x_vals)[:, 1]
plt.plot(x_vals, probs, color='purple')
plt.scatter(X, y, color='black')
plt.xlabel("Exam Score")
plt.ylabel("Pass Probability")
plt.title("Logistic Regression")
plt.grid(True)
plt.show()

# Inference
print("\nInference:")
print("The logistic regression model separates the pass/fail decision well around score 50.")
print("With high accuracy and precision, it’s effective for binary classification in univariate datasets.")
