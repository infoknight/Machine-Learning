#!/usr/bin/env python3

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset and Segregating Independent & Dependent Variables
dataset = pd.read_csv("Data/5_Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Taking care of columns with missing data
#NA

#Encoding Categorical Data
#NA

#Splitting dataset into Training & Testing set
#NA

#Feature Scaling to normalise data
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
ss_y = StandardScaler()
X_scaled = ss_X.fit_transform(X)
y_scaled = ss_y.fit_transform(y.reshape(-1, 1))

#Fitting the SVR Model to the dataset
from sklearn.svm import SVR
regressor = SVR(kernel = "rbf", epsilon = 0.1)

#Predicting the Result & Unscaling
y_pred = regressor.fit(X_scaled, y_scaled)
ans = ss_y.inverse_transform(regressor.predict(X_scaled))

#Visualising the SVR result
'''
plt.scatter(X, y, color = "red")
plt.plot(X, ans, color = "blue")
plt.title("Prediction using SVR")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()
'''

#Predicting the real world user queries
userInput = input("Enter the Value : ")     #Get the user input
userInput = np.array([[userInput]])         #Convert the user input into np array
uI_scaled = ss_X.transform(userInput)       #Apply featue scaling to user input
uI_pred = regressor.predict(uI_scaled)      #Predict the result
ml_pred = ss_y.inverse_transform(uI_pred)   #Unscale the result
print("Resut : %s\n" %float(ml_pred))

