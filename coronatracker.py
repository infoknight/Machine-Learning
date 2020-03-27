#!/usr/bin/env python3

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Data and Segregating Independent & Dependent Variables
dataset = pd.read_csv("Data/coronovirus.csv")
X = dataset.iloc[:, :-1].values
X1 = X
y = dataset.iloc[:, -1].values
#print(dataset)
#print(X)
#print(y)

#Taking Care of Columns with Missing Data
#Not Applicable

#Encoding Categorical Data (Only LabelEncoding is adequate)
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
labelencoder_Date = LabelEncoder()
labelencoder_Date = labelencoder_Date.fit(X1[:, 0])
X1[:, 0] = labelencoder_Date.transform(X1[:, 0])
#print(X)
#print(y)

#Splitting Dataset into Training & Testing data
#Not Required as dataaset is small

#Feature Scaling to normalise Data
#Depends on the algorithm

#Fitting the Polynomial Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 9)
X_poly = poly_reg.fit_transform(X1)          #Creating polynomial expression with required degrees

#poly_reg.fit(X_poly, y)                     #
lin_reg = LinearRegression()
lin_reg.fit(X_poly, y)

#Visualising the PLR Result
plt.scatter(X1, y, color = "red")
plt.plot(X1, lin_reg.predict(X_poly), color = "blue")
plt.title("CoronaVirus : National Performance Monitor")
plt.xlabel("")
plt.ylabel("New Cases Detected")
plt.show()

#Predicting National Performance
userInput = float(input("Enter the new cases identified : "))   #Get the user input
userInput = np.array([[userInput]])                             #Convert the user input into np array
uI_pred = lin_reg.predict(userInput)
print("Prediction Result : %f" %uI_pred)


