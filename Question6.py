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

#3

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# STEP 1: Create synthetic dataset
data = {
    'Age': [22, 25, 47, 52, 46, 56, 55, 60, 24, 23, 34, 31, 40, 43, 51],
    'Income': ['Low', 'Low', 'High', 'High', 'High', 'High', 'Medium', 'Medium', 'Low', 'Low', 'Medium', 'Medium', 'High', 'Medium', 'High'],
    'Student': ['No', 'No', 'No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'No'],
    'Credit_Rating': ['Fair', 'Excellent', 'Fair', 'Fair', 'Fair', 'Excellent', 'Excellent', 'Fair', 'Fair', 'Excellent', 'Excellent', 'Excellent', 'Fair', 'Fair', 'Excellent'],
    'Buys_Computer': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'No']
}

df = pd.DataFrame(data)

# Save to CSV
#df.to_csv("naive_bayes_dataset.csv", index=False)
#print("Dataset saved to 'naive_bayes_dataset.csv'")


# STEP 2: Load dataset and preprocess
#df = pd.read_csv("naive_bayes_dataset.csv")

# Encode categorical features
le_income = LabelEncoder()
df['Income'] = le_income.fit_transform(df['Income'])      # Low=1, Medium=2, High=0 (varies)

le_student = LabelEncoder()
df['Student'] = le_student.fit_transform(df['Student'])    # No=0, Yes=1

le_credit = LabelEncoder()
df['Credit_Rating'] = le_credit.fit_transform(df['Credit_Rating'])  # Fair=0, Excellent=1

le_buys = LabelEncoder()
df['Buys_Computer'] = le_buys.fit_transform(df['Buys_Computer'])  # No=0, Yes=1


# Features and target
X = df.drop('Buys_Computer', axis=1)
y = df['Buys_Computer']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# STEP 3: Train Naive Bayes Classifier

model = GaussianNB()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)


# STEP 4: Evaluation

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

print("Accuracy Score:", accuracy_score(y_test, y_pred))

# Confusion Matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['No', 'Yes'], yticklabels=['No', 'Yes'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

# STEP 5: Predict on New Test Data
# Format: [Age, Income, Student, Credit_Rating]
# Assume: Income=Medium, Student=Yes, Credit_Rating=Fair

new_data = pd.DataFrame({
    'Age': [28],
    'Income': le_income.transform(['Medium']), # Use the correct encoder for Income
    'Student': le_student.transform(['Yes']),   # Use the correct encoder for Student
    'Credit_Rating': le_credit.transform(['Fair']) # Use the correct encoder for Credit_Rating
})

new_pred = model.predict(new_data)
print("\nNew Prediction (Buys Computer):", 'Yes' if new_pred[0] == 1 else 'No') # Use the correct encoder for the output label if needed for inverse_transform, but here we just print Yes/No based on the predicted integer
