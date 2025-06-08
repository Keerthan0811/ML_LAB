# Probabilistic Supervised Learning - Naive Bayes: Create a dataset and Perform the
# necessary pre-processing steps. Train the model using Naive Bayes Classifier. Give new test
# data and predict the classification output. Analyze and write the inference.

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score, classification_report

# Create synthetic dataset (100 samples, 2 features, 2 classes)
X, y = make_classification(n_samples=100, n_features=2, n_informative=2, n_redundant=0, random_state=0)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Train Naive Bayes model
model = GaussianNB()
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))

# Predict new test data
new_sample = [[0.5, -1.2]]
pred = model.predict(new_sample)[0]
print("Prediction for new sample:", "Class 1" if pred else "Class 0")
