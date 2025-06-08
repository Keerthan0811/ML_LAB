# Design an experiment to investigate the impact of varying the number of trees in a Random
# Forest classifier on its performance for a given dataset. Write Python code to implement the
# Random Forest algorithm with different numbers of trees and evaluate its classification
# performance using appropriate evaluation metrics.


from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

# Load dataset and split
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Vary number of trees
trees = [1, 5, 10, 20, 50, 100]
accuracies, f1s = [], []

for n in trees:
    model = RandomForestClassifier(n_estimators=n, random_state=0)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracies.append(accuracy_score(y_test, y_pred))
    f1s.append(f1_score(y_test, y_pred, average='macro'))

# Plot
plt.plot(trees, accuracies, label='Accuracy')
plt.plot(trees, f1s, label='F1 Score')
plt.xlabel("Number of Trees"); plt.ylabel("Score"); plt.legend(); plt.grid()
plt.title("Random Forest Performance vs Number of Trees")
plt.show()
