# Supervised Learning Algorithms - Linear Regression: Assume the dataset to Create a
# Simple Linear Regression model. Predict the scores on the test data and output RMSE and R
# Squared Score. Include appropriate code snippets to visualize the model. Interpret the result.



from sklearn.linear_model import LinearRegression
from sklearn.datasets import make_regression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import numpy as np
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler

data = {
    'Hours_Studied': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'Exam_Score': [30, 35, 45, 50, 60, 65, 70, 75, 85, 95]
}
df = pd.DataFrame(data)

X = df[['Hours_Studied']]
y = df['Exam_Score']
scaler=StandardScaler()
X_processed=scaler.fit_transform(X)
X_train,X_test,y_train,y_test=train_test_split(X_processed,y,test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)

rmse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f"RMSE: {rmse:.2f}")
print(f"R-squared: {r2:.2f}")
plt.scatter(X_processed, y, color='blue', label='Actual')
plt.plot(X_processed, model.predict(X_processed), color='red', label='Predicted')
plt.title("Simple Linear Regression (Scaled Features)")
plt.xlabel("Standardized Hours Studied")
plt.ylabel("Exam Score")
plt.legend()
plt.show()


# Visualization
plt.scatter(X, y, color='blue', label='Actual')
plt.plot(X, model.predict(X), color='red', label='Regression Line')
plt.xlabel('Hours Studied'); plt.ylabel('Score'); plt.title('Simple Linear Regression')
plt.legend(); plt.grid(); plt.show()
