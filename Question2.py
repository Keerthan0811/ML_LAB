# Write a program to demonstrate the working of the decision tree based ID3 algorithm. Use an
# appropriate data set for building the decision tree and apply this knowledge to classify a new
# sample. Interpret the results. Write the inference/analysis of each output.

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt


iris = load_iris()
X, y = iris.data, iris.target
feature_names, target_names = iris.feature_names, iris.target_names


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

model = DecisionTreeClassifier(criterion='entropy')
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy on test set: {acc:.2f}")

sample = [[5.1, 3.5, 1.4, 0.2]]
pred = model.predict(sample)[0]
print("Predicted class for sample:", target_names[pred])

plt.figure(figsize=(10, 6))
plot_tree(model, feature_names=feature_names, class_names=target_names, filled=True)
plt.title("Decision Tree (ID3 - Iris Dataset)")
plt.show()
