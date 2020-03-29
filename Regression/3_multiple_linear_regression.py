#!/usr/bin/env python

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset and Segregating Independent & Dependent Variables
dataset = pd.read_csv("../Data/3_50Startups.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Taking Care of Columns with Missing Data
#Not Applicable

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_State = LabelEncoder()
labelencoder_State = labelencoder_State.fit(X[:, 3])
X[:, 3] = labelencoder_State.transform(X[:, 3])
column_transformer = ColumnTransformer([("ohe", OneHotEncoder(), [3])], remainder = "passthrough")
X = np.array(column_transformer.fit_transform(X), dtype = np.int)

#Avoiding Dummy Variable Trap
X = X[:, 1:]                #Need not specifically code as MLR algorithm automatically takes care of it

#Splitting Dataset into Training & Testing Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Fitting the Training Set
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#Modelling Prediction Using Training Set
train_pred = regressor.predict(X_train)

#Validating Prediction Using Testing Set
test_pred = regressor.predict(X_test)

#Comparing the Actual Values with the Predicted Values
print("Actual Result for X_test : \n")
print(y_test)
print("\n")
print("Predicted Values for X_test : \n")
print(test_pred)


