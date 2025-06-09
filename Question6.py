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


#2


import pandas as pd
import numpy as np
from sklearn import metrics
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report


df=pd.read_csv("/content/class_problem.csv")
le=LabelEncoder()
color=le.fit_transform(df['color'])
legs=le.fit_transform(df['legs'])
height=le.fit_transform(df['height'])
smelly=le.fit_transform(df['smelly'])
species=le.fit_transform(df['species'])

features=pd.DataFrame({
    "color":color,
    "legs":legs,
    "height":height,
    "smelly":smelly
})
label=species


from sklearn.naive_bayes import GaussianNB

model=GaussianNB()
model.fit(features,label)

new=[0,0,1,0]
pred=model.predict([new])
print("newsample-1: predict",pred)
