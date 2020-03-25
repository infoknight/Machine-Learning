#!/usr/bin/env python

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset and Segregating Independent & Dependent Variables
dataset = pd.read_csv("Data/3_50Startups.csv")
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

'''
#Comparing the Actual Values with the Predicted Values
print("Actual Result for X_test : \n")
print(y_test)
print("\n")
print("Predicted Values for X_test : \n")
print(test_pred)
'''

#Building Optimal Model Using Backward Elimination
import statsmodels.api as sm
##MLR ==> y = b0    + b1.X1 + b2.X2 + .... + bn.Xn
###    ==> y = b0.X0 + b1.X1 + b2.X2 + .... + bn.Xn where X0 = 1
##Insert X0 (a column of 1s in the beginning of X
##This manipulation is automatically taken care of by LinearRegression(). However, statsmodel.formula.api needs this to be done manually.
X = np.append(arr = np.ones((50, 1)).astype(int), values = X, axis = 1)  #Adding intercept i.e column of ones 

##Manually finding the most significant Independent Variables with P < 0.05
'''
##Fitting the model with all possible predictors i.e., columns 0, 1, 2, 3 ...
##X : Original Matrix of Features
##X_optimal : Matrix of Optimal Features after eliminating insignifcant columns using Backward Elimination
X_optimal = X[:, [0, 1, 2, 3, 4, 5]]
regressor_OLS = sm.OLS(endog = y, exog = X_optimal).fit()
regressor_OLS.summary()     #Repeat the last three steps to remove the columns with P > 0.05 till identifying only the most significant I.Vs
'''

#Automatically finding the most significant Independent Variables with P < 0.05
def backwardElimination(sl, x_opt):
    numColumns = len(x_opt[0])          #Number of Columns
    for i in range(0, numColumns):
        regressor_OLS = sm.OLS(endog = y, exog = x_opt).fit()
        maxPvalue = max(regressor_OLS.pvalues).astype(float)

        if maxPvalue > sl:
            for j in range(0, numColumns - i):
                if(regressor_OLS.pvalues[j].astype(float) == maxPvalue):
                        x_opt = np.delete(x_opt, j, 1)
            print(regressor_OLS.summary())
    return x_opt

sig_level = 0.05
X_optimal = X[:, [0, 1, 2, 3, 4, 5]]
X_modelled = backwardElimination(sig_level, X_optimal)


#Build automatic Backward Elimination Model using P-values and Adjusted R squares values
print("\n\n")
print("***Build Automatic Backward Elimination Model Using P-values and Adjusted-R-Squares values***")
