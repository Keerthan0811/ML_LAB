# Write a program to demonstrate the working of the decision tree based ID3 algorithm. Use an
# appropriate data set for building the decision tree and apply this knowledge to classify a new
# sample. Interpret the results. Write the inference/analysis of each output.

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt

# Load dataset
iris = load_iris()
X, y = iris.data, iris.target
feature_names, target_names = iris.feature_names, iris.target_names

# Split into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train Decision Tree using ID3 (entropy)
model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {acc:.2f}")

# Predict a new sample (e.g., sepal len=5.1, width=3.5, petal len=1.4, width=0.2)
sample = [[5.1, 3.5, 1.4, 0.2]]
pred = model.predict(sample)[0]
print("Predicted class for sample:", target_names[pred])

# Visualize the tree
plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=feature_names, class_names=target_names, filled=True)
plt.title("Decision Tree (ID3 - Iris Dataset)")
plt.show()
