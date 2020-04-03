#!/usr/bin/env python3
#Assciation Rule Learning : Apriori Model
#This Model requires "apyori_library.py" to be available in the same folder.
#Input to apriory method is a List of Lists

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv("../Data/17_Market_Basket_Optimisation.csv", header = None)
#print(dataset)

#Preparing the List of Lists for all Transactions
transactions = []                           #Initialise of List of Transactions

rows = dataset.shape[0]                     #Number of Rows = Number of Transactions
columns = dataset.shape[1]                  #Number of Columns = Number of Items
#print(numTransactions)
#print(numItems)

for i in range(0, rows):
    transactions.append([str(dataset.values[i, j]) for j in range(0, columns)])
#print(transactions)

#Training apyori Library on the Dataset
from apyori_library import apriori
rules = apriori(transactions, min_support = 0.003, min_confidence = 0.2, min_lift = 3, min_length = 2)
                #min_support:   Let us consider an item that is bought atleat thrice a day for a week over the period of total transa                                ctions (ie., 7501) ==> 3 * 7 / 7500 = 0.003

rules = list(rules)                         #Create a list of rules

#Beautifully Print the Rules using a for Loop
for rule in rules:
    print(rule)
