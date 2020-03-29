#!/usr/bin/env python3

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Data and Segregating Independent & Dependent Variables
dataset = pd.read_csv("Data/corona_newcases.csv")
X = dataset.iloc[:, 0:1].values         #Serial Number
y_Italy = dataset.iloc[:, 2:3].values   #Italy
y_India = dataset.iloc[:, -1].values   #India

#Taking Care of Columns with Missing Data
#Not Applicable

#Encoding Categorical Data (Only LabelEncoding is adequate)
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
labelencoder_Date = LabelEncoder()
labelencoder_Date = labelencoder_Date.fit(X[:, 0])
X[:, 0] = labelencoder_Date.transform(X[:, 0])

#X_poly = poly_reg.fit_transform(X)                  #Creating polynomial expression with required degrees
#Splitting Dataset into Training & Testing data
#Not Required as dataaset is small

#Feature Scaling to normalise Data
from sklearn.preprocessing import StandardScaler
ss_y = StandardScaler()
#print(y_Italy)
#print(y_India)
#y_Italy = ss_y.fit_transform(y_Italy)
#y_India = ss_y.fit_transform(y_India)

#Fitting the Polynomial Linear Regression to the dataset
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
poly_reg = PolynomialFeatures(degree = 3)
X_poly = poly_reg.fit_transform(X)                  #Creating polynomial expression with required degrees
#X_poly_Italy = poly_reg.fit_transform(y_Italy)                  #Creating polynomial expression with required degrees
#X_poly_India = poly_reg.fit_transform(y_India)                  #Creating polynomial expression with required degrees

#Italy
lin_reg_Italy = LinearRegression()
lin_reg_Italy.fit(X_poly, y_Italy)     #Italy
#India
lin_reg_India = LinearRegression()
lin_reg_India.fit(X_poly, y_India)     #India

#Predicting the Result
pred_Italy = lin_reg_Italy.predict(X_poly)
pred_India = lin_reg_India.predict(X_poly)

#Visualising the PLR Result
fig, (ax1, ax2) = plt.subplots(2)
#Italy
ax1.scatter(X, y_Italy, color = "red")
ax1.plot(X, pred_Italy, color = "blue")
ax1.set_title("CoronaVirus : Cases in Italy")
#ax1.set_xlabel("Days")
ax1.set_ylabel("New Cases Detected")
#India
ax2.scatter(X, y_India, color = "red")
ax2.plot(X, pred_India, color = "green")
ax2.set_title("CoronaVirus : Cases in India")
ax2.set_xlabel("Days")
ax2.set_ylabel("New Cases Detected")
plt.show()

'''
#Predicting National Performance
userInput = float(input("Enter the new cases identified : "))   #Get the user input
userInput = np.array([[userInput]])                             #Convert the user input into np array
uI_pred = lin_reg.predict(userInput)
print("Prediction Result : %f" %uI_pred)
'''

