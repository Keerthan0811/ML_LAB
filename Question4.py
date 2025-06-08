# Supervised Learning Algorithms - Linear Regression: Assume the dataset to Create a
# Simple Linear Regression model. Predict the scores on the test data and output RMSE and R
# Squared Score. Include appropriate code snippets to visualize the model. Interpret the result.



import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

# Sample data (e.g., Study Hours vs Scores)
X = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]).reshape(-1, 1)
y = np.array([10, 20, 25, 35, 45, 55, 60, 75, 85, 95])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train Linear Regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"RMSE: {rmse:.2f}, R²: {r2:.2f}")

# Visualization
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Hours Studied'); plt.ylabel('Score'); plt.title('Simple Linear Regression')
plt.legend(); plt.grid(); plt.show()
