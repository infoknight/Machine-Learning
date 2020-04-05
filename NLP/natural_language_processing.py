#!/usr/bin/env python3
#Natural Language Processing

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv("../Data/20_Restaurant_Reviews.tsv", delimiter = "\t", quoting = 3)   #quoting = 3 --> Ignore double quotes
#print(dataset)

#Cleaning the Text
import re
import nltk                                 #NLP Toolkit
nltk.download("stopwords")                  #stopwords package that contains articles, prepositions etc that are irrelevant
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer  #Packaget to stem the words : i.e., loved -> love; likes -> like 

corpus = []                                #dataset of all cleaned reviews
for i in range(0, dataset.shape[0]):        
    review = re.sub('[^a-zA-Z]', ' ', dataset['Review'][i])         #Token Pattern: Remove non-alphabets (punctuations) and insert space (' ' )
    review = review.lower()                                         #Convert to lower case
    review = review.split()                                         #Create a list of words by splitting sentences
    #review = [word for word in review if not word in set(stopwords.words("english"))] #Remove the irrelevant words viz., this, is,.
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words("english"))] #Stemming the words to their root words
    review = ' '.join(review)               #Join the words in the List to form a sentence
    #print(review)
    corpus.append(review)                  #Add the cleaned review to the corpus    
#print(corpus)

#Creating the Bag of Words Model : (Sparse Matrix)
from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)   #Use max_features to limit the column values
X = cv.fit_transform(corpus).toarray()      #X = Matrix of Independent Variables
                                            #convert corpus to matrix using .toarray()
#print(X.shape)                              #Sparse Matrix ; Use max_features = 1500 to limit the column values
y = dataset.iloc[:, 1].values               #Dependent Variable


##Applying Classification Algorithm
#Splitting Dataset into Training & Testing Set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.20, random_state = 0)
#print(X_train)
#print(X_test)
#print(y_train)
#print(y_test)

#Fitting Naive Bayes Classifier to Training Set
from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(X_train, y_train)
#print(classifier)

#Predicting Naive Bayes Result
y_pred = classifier.predict(X_test)

#Confirming the Prediction Accuracy using confusion_matrix
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred)
#print(cm)
correct_0 = cm[0][0]
correct_1 = cm[1][1]
incorrect_0 = cm[0][1]
incorrect_1 = cm[1][0]
accuracy_rate = (correct_0 + correct_1) / X_test.shape[0]
error_rate = (incorrect_0 + incorrect_1) / X_test.shape[0]
print("Correct Predictions of Negative Review : %d" %correct_0)
print("Correct Predictions of Positive Review : %d" %correct_1)
print("Incorrect Predictions of Negative Review : %d" %incorrect_0)
print("Incorrect Predictions of Positive Review : %d" %incorrect_1)
print("Accuracy Rate : %f" %accuracy_rate)
print("Error Rate : %f" %error_rate)
