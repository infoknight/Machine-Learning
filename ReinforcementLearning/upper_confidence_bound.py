#!/usr/bin/env python3
#Reinforcement Learning : Upper Confidence Bound Algorithm
#Additional Reading : https://www.udemy.com/course/machinelearning/learn/#questions/4816936


#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv("../Data/18_Ads_CTR_Optimisation.csv")
#print(dataset)

#Implementing UCB Algorithm from scratch
#Step 1:
#At each round, consider two numbers for each ad.
#Number of times the ad was selected    : number_ad_selected
#Sum of rewards for the ad              : total_reward
num_rounds = 10000
num_ads = 10
ads_selected = []
number_ad_selected = [0] * num_ads          #Initialise an ad_selected counter for each ad
sums_of_reward = [0] * num_ads              #Initialise a net reward counter for each ad
total_reward = 0

#Step 2:
#Compute Average Reward for each ad                 : avg_reward = total_reward / number_ad_selected
#Compute Upper Confidence Bound                     : avg_reward + Delta
#Compute Delta for Confidence Interval for each ad  : 
import math 
for rnd in range(0, num_rounds):            #For each round
    ad = 0
    max_upper_bound = 0
    for i in range(0, num_ads):             #For each ad 
        if (number_ad_selected[i] > 0):
            avg_reward = sums_of_reward[i] / number_ad_selected[i]
            delta_i = math.sqrt(3/2 * math.log(rnd + 1)/number_ad_selected[i])       # rnd + 1 as rnd starts from 0
            upper_bound = avg_reward + delta_i          #Upper Bound for Confidence level
        else:
            upper_bound = 1e400             #10**400
        #Step 3:
        #Select the ad that has maximum Upper Bound         : max(avg_reward + delta)
        if upper_bound > max_upper_bound:
            max_upper_bound = upper_bound
            ad = i
    ads_selected.append(ad)
    number_ad_selected[ad] = number_ad_selected[ad] + 1
    reward = dataset.values[rnd, ad]			#Data from simulated dataset; real-world appln get from online clicks
    sums_of_reward[ad] = sums_of_reward[ad] + reward
    total_reward = total_reward + reward


#print("Total Reward : %d" %total_reward)
#print("Ad Selected : ")
#print(ads_selected)

#Visualising the Result & Selecting the Best Ad
#plt.hist(ads_selected)
plt.bar(range(0, num_ads), number_ad_selected)
plt.title("Reinforcement Learning : Upper Confidence Bound Algorithm")
plt.xlabel("Ads")
plt.ylabel("Number of times each Ad was selected")
plt.show()

