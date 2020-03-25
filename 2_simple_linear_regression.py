#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 24 19:29:47 2020

@author: root
"""
#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import Dataset & Segregate Independent and Dependent Variables
dataset = pd.read_csv("Data/2_Salary_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Taking Care of Columns with Missing Data
##Not Applicable here as there are no missing data##

#Encoding Categorical Data
##Not Applicable##

#Splitting Dataset into Training & Testing Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)

#Feature Scaling to Normalise Dataset
##Not Applicable as Simple Linear Regression Algorithm Automatically Takes Care of it

#Fitting Simple Linear Regression to Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Modelling the Prediction Using Training Set
train_pred = regressor.predict(X_train)

#Validating Model on Testing Set
test_pred  = regressor.predict(X_test)

#Visualising the Model based on Training Set
plt.scatter(X_train, y_train, color = "red")
plt.plot(X_train, train_pred, color = "blue")
plt.title("Experience vs Salary (Training Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary in $")
plt.show()

#Visualising the Model based on Testing Set
plt.scatter(X_test, y_test, color = "red")
plt.plot(X_train, train_pred, color = "blue")
plt.title("Experience vs Salary (Testing Set)")
plt.xlabel("Years of Experience")
plt.ylabel("Salary in $")
plt.show()



