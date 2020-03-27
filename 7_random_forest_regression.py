#!/usr/bin/env python3

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset and Segregating Independent & Dependent Variables
dataset = pd.read_csv("Data/7_Position_Salaries.csv")
X = dataset.iloc[:, 1:2].values
y = dataset.iloc[:, 2].values

#Taking Care of Columns with Missing Data
#Not Applicable

#Encoding Categorical Data
#Not Applicable as Position is already encoded as Level

#Splitting dataset into Training & Testing set
#Not Required as the dataset is already very small

#Feature Scaling to Normalise data
#Depends on the ML algorithm if it incudes this by default

#Fitting the Random Forest Regression to dataset
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators = 300, random_state = 0)
regressor.fit(X, y)

#Visualising the Random Forest Regression Result with High Resolution
X_grid = np.arange(min(X), max(X), 0.01)
X_grid = X_grid.reshape(len(X_grid), 1)
plt.scatter(X, y, color = "red")
plt.plot(X_grid, regressor.predict(X_grid), color = "blue")
plt.title("Random Forest Regression")
plt.xlabel("Position")
plt.ylabel("Salary")
plt.show()

#Prediction the Result for real world applciation
userInput = float(input("Enter the Position : "))   #Get the user input as float
userInput = np.asmatrix(userInput)                  #Convert the user input to matrix
y_pred = regressor.predict(userInput)               #Predict the Salary 
#                                                   #Feature scaling if applicable        
print("Predicted Salary : %f\n" %y_pred)            #Print the prediction

