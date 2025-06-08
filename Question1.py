#Model Measurement Analysis: Using any dataset, calculate TP, TN, FP ,FN and different
#metrics (Accuracy, Precision, Recall(Sensitivity), F1-Score, MCC, Specificity, Negative
#Predictive Value) by defining your own functions. Compare your values with scikit-learn's
# library functions. Get the result of Confusion Matrix using sklearn. Using sklearn, plot the
# ROC & AUC Curves for your test data and random probabilities. Using sklearn, calculate the
# AUC of your test data and of random probabilities. Interpret the results. Write the
# inference/analysis of each output. 


import numpy as np
from sklearn.metrics import *
import matplotlib.pyplot as plt

# Sample data
y_true = [0,1,1,0,1,0,1,0,1,0]
y_pred = [0,0,1,0,0,0,1,1,1,0]
y_prob = [0.3,0.7,0.8,0.2,0.4,0.3,0.9,0.6,0.8,0.1]
rand_prob = np.random.rand(len(y_true))

# Manual metrics
TP = sum((yt==1 and yp==1) for yt, yp in zip(y_true, y_pred))
TN = sum((yt==0 and yp==0) for yt, yp in zip(y_true, y_pred))
FP = sum((yt==0 and yp==1) for yt, yp in zip(y_true, y_pred))
FN = sum((yt==1 and yp==0) for yt, yp in zip(y_true, y_pred))

acc = (TP+TN)/(TP+TN+FP+FN)
prec = TP/(TP+FP) if TP+FP else 0
rec = TP/(TP+FN) if TP+FN else 0
f1 = 2*prec*rec/(prec+rec) if prec+rec else 0
spec = TN/(TN+FP) if TN+FP else 0
npv = TN/(TN+FN) if TN+FN else 0
mcc = (TP*TN - FP*FN) / ((TP+FP)*(TP+FN)*(TN+FP)*(TN+FN))**0.5

print("Manual → Acc: %.2f, Prec: %.2f, Rec: %.2f, F1: %.2f, MCC: %.2f, Spec: %.2f, NPV: %.2f" %
      (acc, prec, rec, f1, mcc, spec, npv))

# Sklearn comparison
print("Sklearn →", "Acc:", accuracy_score(y_true, y_pred),
      "Prec:", precision_score(y_true, y_pred),
      "Rec:", recall_score(y_true, y_pred),
      "F1:", f1_score(y_true, y_pred),
      "\nConf Mat:\n", confusion_matrix(y_true, y_pred))

# ROC & AUC
fpr1, tpr1, _ = roc_curve(y_true, y_prob)
fpr2, tpr2, _ = roc_curve(y_true, rand_prob)
plt.plot(fpr1, tpr1)
plt.plot(fpr2, tpr2, '--',)
plt.plot([0,1],[0,1],'k--');
plt.xlabel("FPR");
plt.ylabel("TPR");
plt.legend();
plt.grid();
plt.show();
