#!/usr/bin/env python3

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset and Segregating Independent & Dependent Variables
dataset = pd.read_csv("../Data/13_Social_Network_Ads.csv")
X = dataset.iloc[:, [2, 3]].values
y = dataset.iloc[:, 4].values
#print(dataset)
#print(X)
#print(y)

#Taking Care of Column with Missing Data
#Not Applicable

#Encoding Categorical Data
#Not Applicable

#Splitting Dataset into Training & Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25, random_state = 0)
#print(X_train)
#print(X_test)
#print(y_train)
#print(y_test)

#Feature Scaling to Normalise Data
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.fit_transform(X_test)
#print(X_train)
#print(X_test)

#Fitting Decision Tree Classifier to Training Set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion = "entropy", random_state = 0)
classifier.fit(X_train, y_train)
#print(classifier)

#Prediction the DTC Result
y_pred = classifier.predict(X_test)
#print(y_pred)

#Confirming the Prediction Accuracy using confusion_matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#print(cm)


#Visualising the Training Result
from matplotlib.colors import ListedColormap
x_set, y_set = X_train, y_train

#Create Meshgrid
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() -1, stop = x_set[:, 0].max() +1 , step = 0.01),\
                     np.arange(start = x_set[:, 1].min() -1, stop = x_set[:, 1].max() +1, step = 0.01))

#Prepare Contour
plt.contour(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,\
            cmap = ListedColormap(("red","green")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

#Colorcode Results
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(("red","green"))(i), label = j)

plt.title("Decision Tree Classifier : Training Result")
plt.legend()
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()

#Visualising the Test Result
from matplotlib.colors import ListedColormap
x_set, y_set = X_train, y_train

#Create Meshgrid
X1, X2 = np.meshgrid(np.arange(start = x_set[:, 0].min() -1, stop = x_set[:, 0].max() +1 , step = 0.01),\
                     np.arange(start = x_set[:, 1].min() -1, stop = x_set[:, 1].max() +1, step = 0.01))

#Prepare Contour
plt.contour(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape), alpha = 0.75,\
            cmap = ListedColormap(("red","green")))
plt.xlim(X1.min(), X1.max())
plt.ylim(X2.min(), X2.max())

#Colorcode Results
for i, j in enumerate(np.unique(y_set)):
    plt.scatter(x_set[y_set == j, 0], x_set[y_set == j, 1], c = ListedColormap(("red","green"))(i), label = j)

plt.title("Decision Tree Classifier : Test Result")
plt.legend()
plt.xlabel("Age")
plt.ylabel("Salary")
plt.show()

