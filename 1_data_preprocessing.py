#!/usr/bin/env python

#Import Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Import Dataset & Segregate Independent and Dependent Variables
dataset = pd.read_csv("Data/1_Data.csv")
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

#Taking Care of Columns with Missing Values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(missing_values = np.nan, strategy = "mean")
imputer = imputer.fit(X[:, 1:3])
X[:, 1:3] = imputer.transform(X[:, 1:3])

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_Country = LabelEncoder()
labelencoder_Country = labelencoder_Country.fit(X[:, 0])
X[:, 0] = labelencoder_Country.transform(X[:, 0])
column_transformer = ColumnTransformer([("ohe",OneHotEncoder(), [0])], remainder = "passthrough")
X = np.array(column_transformer.fit_transform(X),dtype = np.float)

'''
labelencoder_Purchase = LabelEncoder()
labelencoder_Purchase = labelencoder_Purchase.fit(y)
y = labelencoder_Purchase.transform(y)
'''

#Splitting the Dataset into Training & Testing Dataset
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Feature Scaling to Normalise Dataset
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.fit_transform(X_test)

print(X_train)
print(X_test)

'''
ss_y = StandardScaler()
y_train = ss_y.fit_transform(y_train)
y_test = ss_y.fit_transform(y_test)
print(y_train)
print(y_test)
'''




