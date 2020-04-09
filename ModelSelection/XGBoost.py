#!/usr/bin/env python3
#XGBoost for Faster Performance

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#--------------------------------------DATA PREPROCESSING---------------------------------------------------------------------------#
#Importing Dataset and Segregating Independent & Dependent Variables
dataset = pd.read_csv("../Data/28_Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values       #Matrix of Independent Variables
y = dataset.iloc[:, -1].values          #Vector of Dependent Variables
#print(dataset)
#print(X)
#print(y)

#Taking Care of Missing Columns
#Not Applicable

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_Geography = LabelEncoder()
X[:, 1] = labelencoder_Geography.fit_transform(X[:, 1])
labelencoder_Gender = LabelEncoder()
X[:, 2] = labelencoder_Gender.fit_transform(X[:, 2])
columntransformer = ColumnTransformer([("ohe", OneHotEncoder(), [1])], remainder = "passthrough")
X = np.array(columntransformer.fit_transform(X))
#print(X)

#Avoiding Dummy Variable Trap
X = X[:, 1:]
#print(X)

#Splitting Dataset into Training & Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#print(X_train)
#print(X_test)
#print(y_train)
#print(y_test)

#Fitting XGBoost to the Training Set
from xgboost import XGBClassifier
classifier = XGBClassifier(max_depth = 3, learning_rate = 0.1, n_estimators = 100, gamma = 0)
classifier.fit(X_train, y_train)


#--------------------------------------Predicting the Result & Determining Accuracy-------------------------------------------------#
#Predicting the Test Set Results
y_pred = classifier.predict(X_test)     #y_pred is a floating point number 

#Confirming the Accuracy of Prediction Using Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)

#--------------------------------------Applying K-Fold Cross Validation-------------------------------------------------------------#
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = classifier, X = X_train, y = y_train, cv = 10)
print(accuracies)
print("Avg Accuracy : %f" %accuracies.mean())
print("Standard Deviation : %f" %accuracies.std())

