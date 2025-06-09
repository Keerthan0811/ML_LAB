# Build a KNN model for predicting if a person will have diabetes or not with a high accuracy
# score. Perform some appropriate Pre-Processing steps on the given dataset for better results.
# Implement the KNN algorithm on your own. Try other possible processes that can be done to
# dataset and tuning the model to increase accuracy such as Increase K value, Normalization and
# Different Distance Metrics. 



import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('/content/diabetes1.csv')

df_copy=df.copy(deep=True)
df_copy=df_copy[df_copy['Age'].notnull()]
df_copy=df_copy[df_copy['Pregnancies'].notnull()]
df_copy=df_copy[df_copy['BloodPressure'].notnull()]
df_copy=df_copy[df_copy['SkinThickness'].notnull()]
df_copy=df_copy[df_copy['Insulin'].notnull()]
df_copy=df_copy[df_copy['Glucose'].notnull()]
df_copy=df_copy[df_copy['BMI'].notnull()]
df_copy=df_copy[df_copy['DiabetesPedigreeFunction'].notnull()]
df_copy=df_copy[df_copy['Outcome'].notnull()]
print(f"Original Dataset: {df.shape}----- After removing rows: {df_copy.shape}")


zero_na=['Glucose','BloodPressure','SkinThickness','BMI','Insulin']
for col in zero_na:
  df_copy[col]=df_copy[col].replace(0,np.nan)
p=df_copy.hist(figsize=(20,20))


df_copy['Glucose'] = df_copy['Glucose'].fillna(df_copy['Glucose'].mean())
df_copy['BloodPressure'] = df_copy['BloodPressure'].fillna(df_copy['BloodPressure'].mean())
df_copy['SkinThickness'] = df_copy['SkinThickness'].fillna(df_copy['SkinThickness'].median())
df_copy['BMI'] = df_copy['BMI'].fillna(df_copy['BMI'].median())
df_copy['Insulin'] = df_copy['Insulin'].fillna(df_copy['Insulin'].median())
df_copy.head()

sc=StandardScaler()
X=pd.DataFrame(sc.fit_transform(df_copy.drop(['Outcome'],axis=1),),columns=['Pregnancies','Glucose','BloodPressure','SkinThickness','Insulin','BMI','DiabetesPedigreeFunction','Age'])
y=df_copy['Outcome']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


from sklearn.neighbors import KNeighborsClassifier
test_scores=[]
train_scores=[]

for i in range(1,15):
  knn=KNeighborsClassifier(i)
  knn.fit(X_train,y_train)

  train_scores.append(knn.score(X_train,y_train))
  test_scores.append(knn.score(X_test,y_test))



plt.plot(range(1, 15), train_scores, label='Train Scores')
plt.plot(range(1, 15), test_scores, label='Test Scores')
plt.xlabel('i')
plt.ylabel('Score')
plt.legend()
plt.show()







#2
import pandas as pd
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.neighbors import KNeighborsClassifier

# Load dataset from sklearn
data = fetch_openml(name="diabetes", version=1, as_frame=True)
df = data.frame


# Features and label
X = df.drop('class', axis=1)
y = df['class'].map({'tested_negative': 0, 'tested_positive': 1})

# Normalize features
scaler = MinMaxScaler()
X_scaled = scaler.fit_transform(X)

# Split into train-test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Try different K values and metrics
for k in [3, 5, 7]:
    for metric in ['euclidean', 'manhattan']:
        model = KNeighborsClassifier(n_neighbors=k, metric=metric)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        print(f"K={k}, Metric={metric}, Accuracy={acc:.4f}")
