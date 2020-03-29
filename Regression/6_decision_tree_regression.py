#!/usr/bin/env python3

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset and Segregating Independent & Dependent Variables
dataset = pd.read_csv("../Data/6_Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Taking Care of Columns with Missing Data
#Not Applicable

#Encoding Categorical Data
#Not Applicable as Level indicates Positions in the dataset

#Splitting dataset into Training & Testing set
#Not Applicable as dataset is too small. Splitting would result in accuracy issues

#Feature Scaling to normalise data
#Not Required 

#Fitting the DTR to the dataset
from sklearn.tree import DecisionTreeRegressor
regressor = DecisionTreeRegressor(criterion = "mse", random_state = 0)
regressor.fit(X, y)

#Visualising the DecisionTreeRegressor Result with Higher Resoultuion
'''
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid),1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, regressor.predict(X_grid), color = "blue") 
plt.title("Decision Tree Regression")
plt.xlabel("Positon")
plt.ylabel("Salary")
plt.show()
'''

#Predicting the Result for real world application
userInput = float(input("Enter the Position Level : "))     #Get the user input as float
userInput = np.asmatrix(userInput)                          #Convert the user input to matrix value
y_pred = regressor.predict(userInput)                       #Prediction Model
print("Predicted Salary : %f\n" %y_pred)                    #Display the result as float

