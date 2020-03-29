#!/usr/bin/env python3
#Based on 4_polynomial_regression.py
#This program is fine tuned to predict and display the result based on user input as in a real world application

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset and Segregating Independent and Dependent Variables
dataset = pd.read_csv("../Data/4_Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values         # columns[1:2] --> column[1]; However, the result is a Matrix not a Vector
y = dataset.iloc[:, 2].values

#Taking Care of Columns with Missing Data
#Not Applicable

#Encoding Categorical Data
#Not Applicable as Position is already encoded with Level

#Splitting Dataset into Testing & Training Set
#Not needed as the dataset is very small. Also the random selection of training data might might affect the accuracy of the test data if the values are close by

#Feature Scaling to Normalise Data
#Not required as the result would be a polynomial expression with exponential outcome and not linear outcome

#Fitting the Regression Model to the complete Dataset as it was not split into Training & Test set
##We will display the graph for both Simple Linear Regression and Polynomial Linear Regression for comparision
###Fitting the SLR to the Dataset
from sklearn.linear_model import LinearRegression
regressor_lin = LinearRegression()
regressor_lin.fit(X, y)

###Fitting the PLR to the Dataset
###Observe how the model changes by varying "degree". Can it be automated to predict the optimum value?
from sklearn.preprocessing import PolynomialFeatures
poly_X = PolynomialFeatures(degree = 4)     #Convert Simple Expression (ie., y = b0 + b1.X1) to Polynomial Exp (ie., y = b0 + b1.X1 + (b2.X1square))
X_withPoly = poly_X.fit_transform(X)

regressor_poly = LinearRegression()
regressor_poly.fit(X_withPoly, y)


#Prediction Engine
userInput = input("Enter the Position Level : ")
userInput = np.asmatrix(userInput)          #Convert the user input to type matrix
pred_SLR = float(regressor_lin.predict(userInput))

userInput = poly_X.fit_transform(userInput) #Fit the user input to the model
pred_PLR = float(regressor_poly.predict(userInput))

print("\nPrediction based on SLR : %s\n" %pred_SLR)
print("Prediction based on PLR : %s\n" %pred_PLR)


'''
#Visualising Linear Regression
plt.scatter(X, y, color = "red")
plt.plot(X, regressor_lin.predict(X), color = "blue")
plt.title("Plotting Simple Linear Regression Model")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

#Visualising Polynomial Regression
plt.scatter(X, y, color = "red")
plt.plot(X, regressor_poly.predict(X_withPoly), color = "blue")
plt.title("Plotting Polynomial Regression Model")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()
'''


