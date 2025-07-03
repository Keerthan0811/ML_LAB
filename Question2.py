# Write a program to demonstrate the working of the decision tree based ID3 algorithm. Use an
# appropriate data set for building the decision tree and apply this knowledge to classify a new
# sample. Interpret the results. Write the inference/analysis of each output.

from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.metrics import *
import matplotlib.pyplot as plt
import numpy as np
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

iris=load_iris()
df=pd.DataFrame(
    np.c_[iris.data,iris.target],
    columns=iris.feature_names+['target']
)

df.dropna(subset=['target'],inplace=True)
f_cols=iris.feature_names
df[f_cols]=df[f_cols].replace(0,np.nan)
X=df.drop(['target'],axis=1)
y=df['target']

imputer= SimpleImputer(missing_values=np.nan, strategy='median')
X_imputed=imputer.fit_transform(X)
X_df_=pd.DataFrame(X_imputed,columns=X.columns)
scaler=StandardScaler()
X_df=scaler.fit_transform(X_df_)

X_train,X_test,y_train,y_test=train_test_split(X_df,y,test_size=0.2, random_state=42)

id3 = DecisionTreeClassifier(criterion="entropy")
id3.fit(X_train, y_train)

y_pred = id3.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy}")

plt.figure(figsize=(12, 8))
plot_tree(id3, filled=True, feature_names=iris.feature_names, class_names=iris.target_names, rounded=True, proportion=True)
plt.show()
