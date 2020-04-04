#!/usr/bin/env python3
#Reinforcement Learning : Thompson Sampling Algorithm

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv("../Data/19_Ads_CTR_Optimisation.csv")
#print(dataset)

#Implementing Thompson Sampling Algorithm from scratch
#At each round, we consider two numbers for each ad
#Number of times the ad got reward 1 upto round n: numbers_ad_rewarded_1
#Number of times the ad got reward 0 upto round n: numbers_ad_rewarded_0
import random

num_rounds = 10000
num_ads = 10
ads_selected = []
total_reward = 0

#Step 1:
numbers_ad_rewarded_1 = [0] * num_ads           
numbers_ad_rewarded_0 = [0] * num_ads

#Step 2:
for rnd in range(0, num_rounds):
    ad = 0
    max_random = 0
    for i in range(0, num_ads):
        random_beta = random.betavariate(numbers_ad_rewarded_1[i] + 1, numbers_ad_rewarded_0[i] + 1)
        if random_beta > max_random:
            max_random = random_beta
            ad = i
    ads_selected.append(ad)
    reward = dataset.values[rnd, ad]        #Data from simulated dataset; real-world appln get from online clicks
    if reward == 1:
        numbers_ad_rewarded_1[ad] = numbers_ad_rewarded_1[ad] + 1
    else:
        numbers_ad_rewarded_0[ad] = numbers_ad_rewarded_0[ad] + 1
    total_reward = total_reward + reward

print("Total Reward : %d" %total_reward)

#Visualising the Result and Selecting the Best Ad
plt.hist(ads_selected)
#plt.bar(range(0, num_ads), numbers_ad_rewarded_1)
plt.title("Reinforcement Learning : Thompson Sampling Algorithm")
plt.xlabel("Ads")
plt.ylabel("Number of times each Ad was selected")
plt.show()
