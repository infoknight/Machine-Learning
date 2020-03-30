#!/usr/bin/env python3

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset and Segregating Independent & Dependent Variables
dataset = pd.read_csv("../Data/10_Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values
y =dataset.iloc[:, 4].values
#print(dataset)
#print(X)
#print(y)

#Taking Care of Columns with Missing Data
#Not Applicable

#Encoding Categorical Data
#Not Applicable

#Splitting Dataset into Training & Testing Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#print(X_train)
#print(X_test)
#print(y_train)
#print(y_test)

#Feature Scaling
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.fit_transform(X_test)
#print(X_train)
#print(X_test)

#Fitting SVM to Training Set
from sklearn.svm import SVC
classifier = SVC(kernel = "linear", random_state = 0)
classifier.fit(X_train, y_train)

#Predicting the Test Set Results
y_pred = classifier.predict(X_test)

#Confusion Matrix to Confirm the Accuracy of Prediction
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#print(cm)

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
plt.title("SVM : Training Set")
plt.xlabel("Age")
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
plt.title("SVM : Test Set")
plt.xlabel("Age")
plt.ylabel("Salary")
plt.legend()
plt.show()
