#!/usr/bin/env python3
#Deep Learning : Artificial Neural Network

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#--------------------------------------DATA PREPROCESSING---------------------------------------------------------------------------#
#Importing Dataset and Segregating Independent & Dependent Variables
dataset = pd.read_csv("../Data/21_Churn_Modelling.csv")
X = dataset.iloc[:, 3:13].values       #Matrix of Independent Variables
y = dataset.iloc[:, -1].values          #Vector of Dependent Variables
#print(dataset)
#print(X)
#print(y)

#Taking Care of Missing Columns
#Not Applicable

#Encoding Categorical Data
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.compose import ColumnTransformer
labelencoder_Geography = LabelEncoder()
X[:, 1] = labelencoder_Geography.fit_transform(X[:, 1])
labelencoder_Gender = LabelEncoder()
X[:, 2] = labelencoder_Gender.fit_transform(X[:, 2])
columntransformer = ColumnTransformer([("ohe", OneHotEncoder(), [1])], remainder = "passthrough")
X = np.array(columntransformer.fit_transform(X))
#print(X)

#Avoiding Dummy Variable Trap
X = X[:, 1:]
#print(X)

#Splitting Dataset into Training & Test Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
#print(X_train)
#print(X_test)
#print(y_train)
#print(y_test)

#Feature Scaling to Normalise Data
from sklearn.preprocessing import StandardScaler
ss_X = StandardScaler()
X_train = ss_X.fit_transform(X_train)
X_test = ss_X.fit_transform(X_test)
#print(X_train)
#print(X_test)

#--------------------------------------ARTIFICIAL NERUAL NETWORK--------------------------------------------------------------------#

#Importing Libraries
import keras
from keras.models import Sequential
from keras.layers import Dense

#Initialising the ANN
classifier = Sequential()

#Adding the Input Layer and the First Hidden Layer
classifier.add(Dense(input_dim = 11, units = 6, kernel_initializer = "uniform", activation = "relu"))
                        #input_dim  : Num of Independent Variables as in X_train i.e., 11
                        #Number of Neurons in Output layer =1 (for binary classification problems)
                        #units : Num of Neurons in Hidden_Layer = (Neruons in input_dim + Neurons in output layer) / 2
                        #activation : Activation Function for Hidden Layer = ReLU (Rectifier Function)
                        #                                     Output Layer = Sigmoid Function
                        #kernel_initializer      : Adjust weights close to 0
        
#Adding the Second Hidden Layer
classifier.add(Dense(units = 6, kernel_initializer = "uniform", activation = "relu"))

#Adding the Ouput Layer
classifier.add(Dense(units = 1, kernel_initializer = "uniform", activation = "sigmoid"))

#Compiling the ANN by Applying Stochastic Gradient Descent
classifier.compile(optimizer = "adam", loss = "binary_crossentropy", metrics = ["accuracy"])
                        #adam       : Stochastic Gradient Descent Optimizer

#Fitting the ANN to the Training Set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 100)


#--------------------------------------Predicting the Result & Determining Accuracy-------------------------------------------------#
#Predicting the Test Set Results
y_pred = classifier.predict(X_test)     #y_pred is a floating point number 
y_pred = (y_pred > 0.5)                 #y_pred == True if y_pred > 0.5
                                        #          False if y_pred < 0.5

#Confirming the Accuracy of Prediction Using Confusion Matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
print(cm)



