#!/usr/bin/env python3
#Random Selection Algorithm for Reinforcement Learning

#Importing Libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#Importing Dataset
dataset = pd.read_csv("../Data/18_Ads_CTR_Optimisation.csv")
#print(dataset)

#Implementing Random Selection Algorithm
import random
num_rounds = 10000
num_ads = 10
ads_selected = []
total_reward = 0

for rnd in range(0, num_rounds):            #Simulating 10000 rounds
    ad = random.randrange(num_ads)          #Simulating ads clicked (out of 10 ads) in each round
    ads_selected.append(ad)         
    reward = dataset.values[rnd, ad]        #Compare simulated-ads-clicke with the dataset. If matches reward = 1 else 0
    total_reward = total_reward + reward    #Sum up the rewards

#print(ads_selected)                         #List of ads selected in each round
#print(reward)
#print(total_reward)

#Visualising the Random Selection Algorithm Result : Histogram
plt.hist(ads_selected)
plt.title("Reinforcement Learning : Random Selection Algorithm")
plt.xlabel("Ads")
plt.ylabel("Number of times ad selected")
plt.show()
