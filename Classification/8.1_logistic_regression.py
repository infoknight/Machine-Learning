#!/usr/bin/env python3
#This script is the amended version of 8_logistic_regression.py to include the columns Gender & Salary
#Gender column is encoded using LabelEncoder

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset and Segregating Independent & Dependent Variables
dataset = pd.read_csv("../Data/8_Social_Network_Ads.csv")
X = dataset.iloc[:, [1,3]].values       #Includes only the Gender & salary column
y = dataset.iloc[:, 4].values
'''
print("dataset")
print(dataset)
print("X")
print(X)
print("y")
print(y)
'''

#Taking Care of Columns with Missing Data
#Not Applicable for this dataset

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder
labelencoder_Gender = LabelEncoder()
X[:, 0] = labelencoder_Gender.fit_transform(X[:, 0].reshape(X.shape[0], 1))

#Splitting Dataset into Training & Testing sets
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = .25, random_state = 0)
'''
print("X_train")
print(X_train)
print("X_test")
print(X_test)
print("y_train")
print(y_train)
print("y_test")
print(y_test)
'''

#Feature Scaling to Normalise Data
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.fit_transform(X_test)
'''
print("X_train")
print(X_train)
print("X_test")
print(X_test)
'''

#Fitting Logistic Regression Classifier to the Training Set
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, y_train)

#Predicting the Test Set Results
y_pred = classifier.predict(X_test)
'''
print("y_pred")
print(y_pred)
'''

#Making Confusion Matrix to Confirm Accuracy of Prediction
from sklearn.metrics import confusion_matrix        #confusion_matrix is a function and not a Class
cm = confusion_matrix(y_test, y_pred)
'''
print("Confusion Matrix")
print(cm)
'''

#Visualising the Training Set Results
from matplotlib.colors import ListedColormap
X_set, y_set = X_train, y_train
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:, 0].max() +1, step = 0.01), \
                     np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1].max() +1, step = 0.01))
plt.contour(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
            cmap = ListedColormap(("red", "green")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(("red", "green"))(i), label = j)
plt.title("Logistic Regression Classifier : Training Set")
plt.xlabel("Gender")
plt.ylabel("Salary")
plt.legend()
plt.show()

#Visualising the Test Set Results
from matplotlib.colors import ListedColormap
X_set, y_set = X_test, y_test
X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() -1, stop = X_set[:, 0].max() +1, step = 0.01), \
                     np.arange(start = X_set[:, 1].min() -1, stop = X_set[:, 1].max() +1, step = 0.01))
plt.contour(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75, 
            cmap = ListedColormap(("red", "green")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1], c = ListedColormap(("red", "green"))(i), label = j)
plt.title("Logistic Regression Classifier : Test Set")
plt.xlabel("Gender")
plt.ylabel("Salary")
plt.legend()
plt.show()
