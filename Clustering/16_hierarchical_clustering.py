#!/usr/bin/env python3

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv("../Data/16_Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values      #Select Annual Income and Spending Score columns
#print(dataset)
#print(X)

#Visualising the Dataset for Appreciation
plt.scatter(X[:, 0], X[:, 1])
plt.title("Mall Dataset")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.show()

#Find K : Number of Clusters Using Dendrogram
import scipy.cluster.hierarchy as sch          # <<== Using Scipy 
dendrogram = sch.dendrogram(sch.linkage(X, method = "ward"))
plt.title("Finding K using Dendrogram")
plt.xlabel("Customers")                        #Not Annual Income 
plt.ylabel("Euclidean Distance")
plt.show()

#K = 5  #as observed from the Dendrogram

#Fitting Hierarchical Clustering to the Dataset & Predicting Clusters
from sklearn.cluster import AgglomerativeClustering
hc = AgglomerativeClustering(n_clusters = 5, affinity = "euclidean", linkage = "ward")
y_pred = hc.fit_predict(X)                      #Predicting the Clusters
#print(y_pred)

'''
#Visualising Clusters after Renaming Labels According to Business Needs
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s = 100, c = "red", label = "Cluster-0")
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s = 100, c = "blue", label = "Cluster-1")
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s = 100, c = "green", label = "Cluster-2")
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], s = 100, c = "yellow", label = "Cluster-3")
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], s = 100, c = "brown", label = "Cluster-4")
plt.title("Clusters of Customers")
plt.legend()
plt.xlabel("Annual Income (K$)")
plt.ylabel("Spending Score (1 - 100)")
plt.show()
'''

#Visualising Clusters
plt.scatter(X[y_pred == 0, 0], X[y_pred == 0, 1], s = 100, c = "red", label = "Target")
plt.scatter(X[y_pred == 1, 0], X[y_pred == 1, 1], s = 100, c = "blue", label = "Standard")
plt.scatter(X[y_pred == 2, 0], X[y_pred == 2, 1], s = 100, c = "green", label = "Luxury")
plt.scatter(X[y_pred == 3, 0], X[y_pred == 3, 1], s = 100, c = "yellow", label = "Spendthrift")
plt.scatter(X[y_pred == 4, 0], X[y_pred == 4, 1], s = 100, c = "brown", label = "Sensible")
plt.title("Clusters of Customers")
plt.legend()
plt.xlabel("Annual Income (K$)")
plt.ylabel("Spending Score (1 - 100)")
plt.show()

