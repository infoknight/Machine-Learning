#!/usr/bin/env python3
#Problem Statement : Find the Clusters based on Annual Income and Spending Score

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset and Segregating Independent & Dependent Variables
dataset = pd.read_csv("../Data/15_Mall_Customers.csv")
X = dataset.iloc[:, [3, 4]].values      #Select Annual Income & Spending Score
#y                                      #y is not required as we need to find only the clusters
#print(dataset)
#print(X)

#Visualising the dataset
#print(X[:, 0])
#print(X[:, 1])
plt.scatter(X[:, 0], X[:, 1])
plt.title("Mall Clients")
plt.xlabel("Annual Income (K$)")
plt.ylabel("Spending Score")
plt.show()

#Find K   : Using the Elbow Method to Find the Optimal Number of Clusters (K)
from sklearn.cluster import KMeans
wcss = []                               #Within Cluster Sum of Squares 
for i in range(1, 11):                  #K = 1 to 10
    kmeans = KMeans(n_clusters = i, init = "k-means++", n_init = 10, max_iter = 300, random_state = 0)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
'''
plt.plot(range(1,11), wcss)
plt.title("The Elbow Method to Find K")
plt.xlabel("Number of Clusters 'K'")
plt.ylabel("WCSS")
plt.show()
'''
#K = 5, as seen from the plot

#Predict the Results
kmeans = KMeans(n_clusters = 5, init = "k-means++", n_init = 10, max_iter = 300, random_state = 0)
y_kmeans = kmeans.fit_predict(X)

#Visualising the K-Means Clusters
'''
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = "red", label = "Cluster-0")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = "blue", label = "Cluster-1")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = "green", label = "Cluster-2")
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = "cyan", label = "Cluster-3")
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = "magenta", label = "Cluster-4")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = "yellow", label = "Centroids")
plt.title("K-Means Clustering : Cluster of Mall Clients")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1 - 100)")
plt.legend()
plt.show()
'''

#Visualising After Renaming Clusters wrt Business Needs
plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 1], s = 100, c = "red", label = "Comfort Seekers")
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 1], s = 100, c = "blue", label = "Spendthrift")
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 1], s = 100, c = "green", label = "Luxury")
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 1], s = 100, c = "cyan", label = "Sensible")
plt.scatter(X[y_kmeans == 4, 0], X[y_kmeans == 4, 1], s = 100, c = "magenta", label = "Targets")
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 1], s = 300, c = "yellow", label = "Centroids")
plt.title("K-Means Clustering : Cluster of Mall Clients")
plt.xlabel("Annual Income (k$)")
plt.ylabel("Spending Score (1 - 100)")
plt.legend()
plt.show()
