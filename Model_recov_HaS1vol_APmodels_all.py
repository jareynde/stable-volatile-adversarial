#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rescorla Wagner model with epsilon-greedy decision making policy
            
Model recovery of 4 models in volatile context,
according to the block structure of Hide and Seek experiment 1 (meaning Q-values reset every 200 trials):

*eps-lr
*eps-lr-uc
*eps-2lr (positive PE lr and negative PE lr)
*eps-2lr-uc

@author: Janne Reynders; janne.reynders@ugent.be
"""
import numpy as np                  
import pandas as pd                 
import matplotlib.pyplot as plt     
import matplotlib
import os
import random
from collections import Counter
import scipy.stats as stats
import itertools
import statsmodels.stats.multitest as smm
import math
from scipy.optimize import differential_evolution # finding optimal params in models


matplotlib.rcParams['font.family'] = 'times new roman'

#Simulations
#simulation of Rescorla-Wagner model in an adversarial context

#Model 1
def RW_eps_lr_volatile(eps, lr, T, Q_int):
    #eps        --->        epsilon
    #lr         --->        learning rate
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)

    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards


    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    
    v = np.random.normal(loc=15, scale=3)
    v = np.zeros(800)
    for i in range(800):
        v[i] = round(np.random.normal(loc=15, scale=3))
    v = np.cumsum(v)
    v=v[v<10000] 
    reward_volatile = [0.1,0.1,0.1,0.1,0.1,0.9,0.9,0.9]
    for t in range(T):

        #reset every length of a block of trials in HaS exp 1
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice

        
        # store values for Q
        Q_k_stored[t,:] = Q_k   
              
        # make choice based on choice probababilities
        rand[t] = np.random.choice([0,1], p=[1-eps,eps])
        if rand[t] == 0:
            k[t] = np.argmax(Q_k)
        if rand[t] == 1:
            k[t] = np.random.choice(range(K))
        
        # volatile
        if t in v:
            min_vol = np.min(reward_volatile)
            max_vol = np.max(reward_volatile)
            index_vol = np.where(reward_volatile == min_vol)[0]
            random.shuffle(index_vol)
            new_max_index = index_vol[0:3]
            for i in range(8):
                if i in new_max_index:
                    reward_volatile[i] = max_vol
                else: reward_volatile[i] = min_vol
        a1 = reward_volatile[k[t]]
        a0 = 1-a1
        r[t] = np.random.choice([1, 0], p=[a1, a0])

         # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + lr * delta_k


    return k, r, Q_k_stored


#Model 2:
def RW_eps_lr_uc_volatile(eps, lr, uc, T, Q_int):
    #eps        --->        epsilon
    #lr         --->        learning rate
    #uc         --->        unchosen value-bias
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)

    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards


    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    
    v = np.random.normal(loc=15, scale=3)
    v = np.zeros(800)
    for i in range(800):
        v[i] = round(np.random.normal(loc=15, scale=3))
    v = np.cumsum(v)
    v=v[v<10000] 
    reward_volatile = [0.1,0.1,0.1,0.1,0.1,0.9,0.9,0.9]    
    for t in range(T):

        #reset every length of a block of trials in HaS exp 1
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
        
        # store values for Q
        Q_k_stored[t,:] = Q_k   
              
        # make choice based on choice probababilities
        rand[t] = np.random.choice([0,1], p=[1-eps,eps])
        if rand[t] == 0:
            k[t] = np.argmax(Q_k)
        if rand[t] == 1:
            k[t] = np.random.choice(range(K))
        
        # volatile
        if t in v:
            min_vol = np.min(reward_volatile)
            max_vol = np.max(reward_volatile)
            index_vol = np.where(reward_volatile == min_vol)[0]
            random.shuffle(index_vol)
            new_max_index = index_vol[0:3]
            for i in range(8):
                if i in new_max_index:
                    reward_volatile[i] = max_vol
                else: reward_volatile[i] = min_vol
        a1 = reward_volatile[k[t]]
        a0 = 1-a1
        r[t] = np.random.choice([1, 0], p=[a1, a0])


         # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + lr * delta_k

        # update Q values for unchosen option:
        Q_k += uc  # Apply the unchosen bias to all Q-values
        Q_k[k[t]] -= uc  # Counteract the bias for the chosen option to keep it balanced

    return k, r, Q_k_stored


#Model 5:
def RW_eps_2lr_volatile(eps, lrpos, lrneg, T, Q_int):
    #eps        --->        epsilon
    #lr         --->        learning rate
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)

    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards


    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    
    v = np.random.normal(loc=15, scale=3)
    v = np.zeros(800)
    for i in range(800):
        v[i] = round(np.random.normal(loc=15, scale=3))
    v = np.cumsum(v)
    v=v[v<10000] 
    reward_volatile = [0.1,0.1,0.1,0.1,0.1,0.9,0.9,0.9]
    for t in range(T):

        #reset every length of a block of trials in HaS exp 1
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
        
        # store values for Q
        Q_k_stored[t,:] = Q_k   
              
        # make choice based on choice probababilities
        rand[t] = np.random.choice([0,1], p=[1-eps,eps])
        if rand[t] == 0:
            k[t] = np.argmax(Q_k)
        if rand[t] == 1:
            k[t] = np.random.choice(range(K))
        
        # volatile
        if t in v:
            min_vol = np.min(reward_volatile)
            max_vol = np.max(reward_volatile)
            index_vol = np.where(reward_volatile == min_vol)[0]
            random.shuffle(index_vol)
            new_max_index = index_vol[0:3]
            for i in range(8):
                if i in new_max_index:
                    reward_volatile[i] = max_vol
                else: reward_volatile[i] = min_vol
        a1 = reward_volatile[k[t]]
        a0 = 1-a1
        r[t] = np.random.choice([1, 0], p=[a1, a0])


         # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        if delta_k < 0: 
            Q_k[k[t]] = Q_k[k[t]] + lrneg * delta_k
        if delta_k >= 0:
            Q_k[k[t]] = Q_k[k[t]] + lrpos * delta_k


    return k, r, Q_k_stored


#Model 6:
def RW_eps_2lr_uc_volatile(eps, lrpos, lrneg, uc, T, Q_int):
    #eps        --->        epsilon
    #lr         --->        learning rate
    #uc         --->        unchosen value-bias
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)

    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards


    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice

    v = np.random.normal(loc=15, scale=3)
    v = np.zeros(800)
    for i in range(800):
        v[i] = round(np.random.normal(loc=15, scale=3))
    v = np.cumsum(v)
    v=v[v<10000] 
    reward_volatile = [0.1,0.1,0.1,0.1,0.1,0.9,0.9,0.9]
    
    
    for t in range(T):

        #reset every length of a block of trials in HaS exp 1
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
        
        # store values for Q
        Q_k_stored[t,:] = Q_k   
              
        # make choice based on choice probababilities
        rand[t] = np.random.choice([0,1], p=[1-eps,eps])
        if rand[t] == 0:
            k[t] = np.argmax(Q_k)
        if rand[t] == 1:
            k[t] = np.random.choice(range(K))
        
        
        # volatile
        if t in v:
            min_vol = np.min(reward_volatile)
            max_vol = np.max(reward_volatile)
            index_vol = np.where(reward_volatile == min_vol)[0]
            random.shuffle(index_vol)
            new_max_index = index_vol[0:3]
            for i in range(8):
                if i in new_max_index:
                    reward_volatile[i] = max_vol
                else: reward_volatile[i] = min_vol
        a1 = reward_volatile[k[t]]
        a0 = 1-a1
        r[t] = np.random.choice([1, 0], p=[a1, a0])

         # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        if delta_k < 0: 
            Q_k[k[t]] = Q_k[k[t]] + lrneg * delta_k
        if delta_k >= 0:
            Q_k[k[t]] = Q_k[k[t]] + lrpos * delta_k

        # update Q values for unchosen option:
        Q_k += uc  # Apply the unchosen bias to all Q-values
        Q_k[k[t]] -= uc  # Counteract the bias for the chosen option to keep it balanced

    return k, r, Q_k_stored




#negative loglikelihoods for each model

#model 1
def negll_RW_eps_lr(params, k, r):

    eps, lr = params
    Q_int = 1
    K = np.max(k)+1
    T = len(k)

    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)

    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):

        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice

        Q_k_stored[t,:] = Q_k
        
        # Compute choice probabilities based on epsilon-greedy policy
        p = np.full(K, eps / K)  # Start with probability eps/K for each option
        max_Q = np.argmax(Q_k)  # Find the option with the maximum Q value
        p[max_Q] += 1 - eps  # Add the (1 - eps) probability to the greedy option

        # Record the probability of the chosen option
        choice_prob[t] = p[k[t]]
        choice_prob[t] = max(p[k[t]], 1e-10)

        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] += lr * delta_k


    negLL = -np.sum(np.log(choice_prob)) 

    return negLL


#model 2
def negll_RW_eps_lr_uc(params, k, r):

    eps, lr, uc = params
    Q_int = 1
    K = np.max(k)+1
    T = len(k)

    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)

    choice_prob = np.zeros((T), dtype = float)
    

    for t in range(T):

        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice


        Q_k_stored[t,:] = Q_k
        
        # Compute choice probabilities based on epsilon-greedy policy
        p = np.full(K, eps / K)  # Start with probability eps/K for each option
        max_Q = np.argmax(Q_k)  # Find the option with the maximum Q value
        p[max_Q] += 1 - eps  # Add the (1 - eps) probability to the greedy option

        # Record the probability of the chosen option
        choice_prob[t] = p[k[t]]
        choice_prob[t] = max(p[k[t]], 1e-10)

        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] += lr * delta_k

        # update Q values for unchosen option:
        Q_k += uc  # Apply the unchosen bias to all Q-values
        Q_k[k[t]] -= uc  # Counteract the bias for the chosen option to keep it balanced

    negLL = -np.sum(np.log(choice_prob)) 

    return negLL


#model 3
def negll_RW_eps_lr_lrs(params, k, r):

    eps, lr, lrs, w = params
    Q_int = 1
    K = np.max(k)+1
    K_seq = K*K
    T = len(k)

    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)

    S_k = np.ones(K_seq)*Q_int
    S_k_stored = np.zeros((T,K_seq), dtype=float)

    seq_options = np.array([[a, b] for a in range(K) for b in range(K)])

    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
            S_k = np.ones(K_seq)*Q_int #initual value of Q for each choice

        Q_k_stored[t,:] = Q_k
        S_k_stored[t,:] = S_k 

        if t == 0:
            next_seq_options = np.zeros(K)

        combined_QandS_info = w*Q_k + (1-w)*next_seq_options
        # Compute choice probabilities based on epsilon-greedy policy
        p = np.full(K, eps / K)  # Start with probability eps/K for each option
        max_C = np.argmax(combined_QandS_info)  # Find the option with the maximum Combined value
        p[max_C] += 1 - eps  # Add the (1 - eps) probability to the greedy option

        # Record the probability of the chosen option
        choice_prob[t] = p[k[t]]
        choice_prob[t] = max(p[k[t]], 1e-10)
        
        if t != 0:
            current_seq = k[t-1:t+1]
            current_index = np.where(np.all(seq_options==current_seq,axis=1))[0]
            # update Q_values of sequences
            sdelta_k = r[t] - S_k[current_index]
            S_k[current_index] = S_k[current_index] + lrs * sdelta_k

            #if the current choice is x, next_seq_options stores the S values of all possible next response pairs (that start with x)
            next_seq_options = np.zeros(K)
            for i in range(K):
                next_seq = [k[t], i]
                next_index = np.where(np.all(seq_options==next_seq,axis=1))[0][0]
                next_seq_options[i] = S_k[next_index]

        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] += lr * delta_k

    negLL = -np.sum(np.log(choice_prob)) 

    return negLL

#model 4
def negll_RW_eps_lr_uc_lrs(params, k, r):

    eps, lr, uc, lrs, w = params
    Q_int = 1
    K = np.max(k)+1
    K_seq = K*K
    T = len(k)

    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)

    S_k = np.ones(K_seq)*Q_int
    S_k_stored = np.zeros((T,K_seq), dtype=float)

    seq_options = np.array([[a, b] for a in range(K) for b in range(K)])

    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
            S_k = np.ones(K_seq)*Q_int #initual value of Q for each choice

        Q_k_stored[t,:] = Q_k
        S_k_stored[t,:] = S_k 

        if t == 0:
            next_seq_options = np.zeros(K)

        combined_QandS_info = w*Q_k + (1-w)*next_seq_options
        # Compute choice probabilities based on epsilon-greedy policy
        p = np.full(K, eps / K)  # Start with probability eps/K for each option
        max_C = np.argmax(combined_QandS_info)  # Find the option with the maximum Combined value
        p[max_C] += 1 - eps  # Add the (1 - eps) probability to the greedy option

        # Record the probability of the chosen option
        choice_prob[t] = p[k[t]]
        choice_prob[t] = max(p[k[t]], 1e-10)
        
        if t != 0:
            current_seq = k[t-1:t+1]
            current_index = np.where(np.all(seq_options==current_seq,axis=1))[0]
            # update Q_values of sequences
            sdelta_k = r[t] - S_k[current_index]
            S_k[current_index] = S_k[current_index] + lrs * sdelta_k

            #if the current choice is x, next_seq_options stores the S values of all possible next response pairs (that start with x)
            next_seq_options = np.zeros(K)
            for i in range(K):
                next_seq = [k[t], i]
                next_index = np.where(np.all(seq_options==next_seq,axis=1))[0][0]
                next_seq_options[i] = S_k[next_index]

        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] += lr * delta_k

        # update Q values for unchosen option:
        Q_k += uc  # Apply the unchosen bias to all Q-values
        Q_k[k[t]] -= uc  # Counteract the bias for the chosen option to keep it balanced


    negLL = -np.sum(np.log(choice_prob)) 

    return negLL

#model 5
def negll_RW_eps_2lr(params, k, r):

    eps, lrpos, lrneg = params
    Q_int = 1
    K = np.max(k)+1
    T = len(k)
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)
    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice

        Q_k_stored[t,:] = Q_k
        
        # Compute choice probabilities based on epsilon-greedy policy
        p = np.full(K, eps / K)  # Start with probability eps/K for each option
        max_Q = np.argmax(Q_k)  # Find the option with the maximum Q value
        p[max_Q] += 1 - eps  # Add the (1 - eps) probability to the greedy option

        # Record the probability of the chosen option
        choice_prob[t] = p[k[t]]
        choice_prob[t] = max(p[k[t]], 1e-10)

         # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        if delta_k < 0: 
            Q_k[k[t]] = Q_k[k[t]] + lrneg * delta_k
        if delta_k >= 0:
            Q_k[k[t]] = Q_k[k[t]] + lrpos * delta_k


    negLL = -np.sum(np.log(choice_prob)) 

    return negLL


#model 6
def negll_RW_eps_2lr_uc(params, k, r):

    eps, lrpos, lrneg, uc = params
    Q_int = 1
    K = np.max(k)+1
    T = len(k)
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)
    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):

        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice

        Q_k_stored[t,:] = Q_k
        
        # Compute choice probabilities based on epsilon-greedy policy
        p = np.full(K, eps / K)  # Start with probability eps/K for each option
        max_Q = np.argmax(Q_k)  # Find the option with the maximum Q value
        p[max_Q] += 1 - eps  # Add the (1 - eps) probability to the greedy option

        # Record the probability of the chosen option
        choice_prob[t] = p[k[t]]
        choice_prob[t] = max(p[k[t]], 1e-10)

         # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        if delta_k < 0: 
            Q_k[k[t]] = Q_k[k[t]] + lrneg * delta_k
        if delta_k >= 0:
            Q_k[k[t]] = Q_k[k[t]] + lrpos * delta_k

        # update Q values for unchosen option:
        Q_k += uc  # Apply the unchosen bias to all Q-values
        Q_k[k[t]] -= uc  # Counteract the bias for the chosen option to keep it balanced


    negLL = -np.sum(np.log(choice_prob)) 

    return negLL


#model 7
def negll_RW_eps_2lr_lrs(params, k, r):

    eps, lrpos, lrneg, lrs, w = params
    Q_int = 1
    K = np.max(k)+1
    K_seq = K*K
    T = len(k)


    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)

    S_k = np.ones(K_seq)*Q_int
    S_k_stored = np.zeros((T,K_seq), dtype=float)

    seq_options = np.array([[a, b] for a in range(K) for b in range(K)])

    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):

        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
            S_k = np.ones(K_seq)*Q_int #initual value of Q for each choice

        Q_k_stored[t,:] = Q_k
        S_k_stored[t,:] = S_k 

        if t == 0:
            next_seq_options = np.zeros(K)
       
        combined_QandS_info = w*Q_k + (1-w)*next_seq_options
        # Compute choice probabilities based on epsilon-greedy policy
        p = np.full(K, eps / K)  # Start with probability eps/K for each option
        max_C = np.argmax(combined_QandS_info)  # Find the option with the maximum Combined value
        p[max_C] += 1 - eps  # Add the (1 - eps) probability to the greedy option

        # Record the probability of the chosen option
        choice_prob[t] = p[k[t]]
        choice_prob[t] = max(p[k[t]], 1e-10)


        if t != 0:
            current_seq = k[t-1:t+1]
            current_index = np.where(np.all(seq_options==current_seq,axis=1))[0]
            # update Q_values of sequences
            sdelta_k = r[t] - S_k[current_index]
            S_k[current_index] = S_k[current_index] + lrs * sdelta_k

            #if the current choice is x, next_seq_options stores the S values of all possible next response pairs (that start with x)
            next_seq_options = np.zeros(K)
            for i in range(K):
                next_seq = [k[t], i]
                next_index = np.where(np.all(seq_options==next_seq,axis=1))[0][0]
                next_seq_options[i] = S_k[next_index]


         # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        if delta_k < 0: 
            Q_k[k[t]] = Q_k[k[t]] + lrneg * delta_k
        if delta_k >= 0:
            Q_k[k[t]] = Q_k[k[t]] + lrpos * delta_k


    negLL = -np.sum(np.log(choice_prob)) 

    return negLL



#model 8
def negll_RW_eps_2lr_uc_lrs(params, k, r):

    eps, lrpos, lrneg, uc, lrs, w = params
    Q_int = 1
    K = np.max(k)+1
    K_seq = K*K
    T = len(k)


    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    Q_k_stored = np.zeros((T,K), dtype=float)

    S_k = np.ones(K_seq)*Q_int
    S_k_stored = np.zeros((T,K_seq), dtype=float)

    seq_options = np.array([[a, b] for a in range(K) for b in range(K)])

    choice_prob = np.zeros((T), dtype = float)
    
    for t in range(T):

        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
            S_k = np.ones(K_seq)*Q_int #initual value of Q for each choice

        Q_k_stored[t,:] = Q_k
        S_k_stored[t,:] = S_k 

        if t == 0:
            next_seq_options = np.zeros(K)
       
        combined_QandS_info = w*Q_k + (1-w)*next_seq_options
        # Compute choice probabilities based on epsilon-greedy policy
        p = np.full(K, eps / K)  # Start with probability eps/K for each option
        max_C = np.argmax(combined_QandS_info)  # Find the option with the maximum Combined value
        p[max_C] += 1 - eps  # Add the (1 - eps) probability to the greedy option

        # Record the probability of the chosen option
        choice_prob[t] = p[k[t]]
        choice_prob[t] = max(p[k[t]], 1e-10)


        if t != 0:
            current_seq = k[t-1:t+1]
            current_index = np.where(np.all(seq_options==current_seq,axis=1))[0]
            # update Q_values of sequences
            sdelta_k = r[t] - S_k[current_index]
            S_k[current_index] = S_k[current_index] + lrs * sdelta_k

            #if the current choice is x, next_seq_options stores the S values of all possible next response pairs (that start with x)
            next_seq_options = np.zeros(K)
            for i in range(K):
                next_seq = [k[t], i]
                next_index = np.where(np.all(seq_options==next_seq,axis=1))[0][0]
                next_seq_options[i] = S_k[next_index]


         # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        if delta_k < 0: 
            Q_k[k[t]] = Q_k[k[t]] + lrneg * delta_k
        if delta_k >= 0:
            Q_k[k[t]] = Q_k[k[t]] + lrpos * delta_k

        # update Q values for unchosen option:
        Q_k += uc  # Apply the unchosen bias to all Q-values
        Q_k[k[t]] -= uc  # Counteract the bias for the chosen option to keep it balanced


    negLL = -np.sum(np.log(choice_prob)) 

    return negLL





def calculate_rmse(actual_values, predicted_values):
    n = len(actual_values)
    squared_diffs = [(actual_values[i] - predicted_values[i]) ** 2 for i in range(n)]
    mean_squared_diff = sum(squared_diffs) / n
    rmse = math.sqrt(mean_squared_diff)
    return rmse
def calculate_Rsquared(y_true, y_pred):
    # Calculate the mean of observed values
    mean_y_true = np.mean(y_true)

    # Calculate the Sum of Squared Residuals (SSR)
    ssr = sum((y_true[i] - y_pred[i])**2 for i in range(len(y_true)))

    # Calculate the Total Sum of Squares (SST)
    sst = sum((y_true[i] - mean_y_true)**2 for i in range(len(y_true)))

    # Calculate R-squared
    Rsquared = 1 - (ssr / sst)
    return Rsquared 


#    eps, lr = params
#    eps, lr, uc = params
#    eps, lr, lrs, w = params
#    eps, lr, uc, lrs, w = params

#    eps, lrpos, lrneg = params
#    eps, lrpos, lrneg, uc = params
#    eps, lrpos, lrneg, lrs, w = params
#    eps, lrpos, lrneg, uc, lrs, w = params


T=800
recov_amount = 50
Q_int = 1

bounds1=[(0,1), (0,1)]
bounds2=[(0,1), (0,1),(-3,3)]

bounds5=[(0,1), (0,1), (0,1)]
bounds6=[(0,1), (0,1), (0,1),(-3,3)]



strategies = ['best1bin','best1exp','rand1exp','rand2bin','rand2exp','randtobest1bin','randtobest1exp','currenttobest1bin','currenttobest1exp','best2exp','best2bin']
strategies = ['randtobest1bin']

script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
next_dir1 = os.path.join(parent_dir, '4Data_fit')
next_dir2 = os.path.join(next_dir1, 'output_AP_all')

### HAS ###
for strategy in strategies:
    print(strategy)

    folder_name = f"output_{strategy}"
    save_dir = os.path.join(script_dir, folder_name)
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)

    
    #########################################################################################
    #simulate model 1 [eps-lr]
    #########################################################################################
    sample_dir = os.path.join(next_dir2, '1_HaS1vol_modelfit_eps_lr.xlsx')
    sample_pool = pd.read_excel(sample_dir)

    M1BIC_all = pd.DataFrame(index=range(0, 1), columns=['eps-lr', 'eps-lr-uc', 'eps-2lr', 'eps-2lr-uc'])
    M1sampled_eps_lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr'])
    M1eps_lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'negLL', 'BIC'])
    M1eps_lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'uc', 'negLL', 'BIC'])
    M1eps_2lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'negLL', 'BIC'])    
    M1eps_2lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'uc', 'negLL', 'BIC'])

    for i in range(recov_amount):
        print('M1:', i+1)
        while True:
            ri = np.random.randint(0, len(sample_pool['eps']))
            true_eps = float(sample_pool['eps'].iloc[ri])
            if true_eps < 0.7:
                break
        ri = np.random.randint(0, len(sample_pool['eps']))
        true_lr = float(sample_pool['lr'].iloc[ri])
        M1sampled_eps_lr.at[i, 'eps'] = true_eps
        M1sampled_eps_lr.at[i, 'lr'] = true_lr
        print('true_eps:', true_eps)
        print('true_lr:', true_lr)

        title_excel = os.path.join(save_dir, f'HaS1vol_M1_modelrecov_eps_lr_sampled.xlsx')
        M1sampled_eps_lr.to_excel(title_excel, index=False)

        k, r, Q_k_stored = RW_eps_lr_volatile(eps=true_eps, lr=true_lr, T=T, Q_int=Q_int)



        #model 1 [eps-lr]
        #eps, lr = params
        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_lr, bounds=bounds1, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds1) * np.log(T) + 2*negLL

        #store in dataframe
        M1eps_lr.at[i, 'eps'] = param_fits[0]
        M1eps_lr.at[i, 'lr'] = param_fits[1]
        M1eps_lr.at[i, 'negLL'] = negLL
        M1eps_lr.at[i, 'BIC'] = BIC

        M1BIC_all.at[i, 'eps-lr'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1vol_M1_1modelrecov_eps_lr.xlsx')
        M1eps_lr.to_excel(title_excel, index=False)



        #model 2 [eps-lr-uc]
        #eps, lr, uc = params
        negLL = np.inf #initialize negative log likelihood

        result = differential_evolution(negll_RW_eps_lr_uc, bounds=bounds2, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds2) * np.log(T) + 2*negLL

        #store in dataframe
        M1eps_lr_uc.at[i, 'eps'] = param_fits[0]
        M1eps_lr_uc.at[i, 'lr'] = param_fits[1]
        M1eps_lr_uc.at[i, 'uc'] = param_fits[2]
        M1eps_lr_uc.at[i, 'negLL'] = negLL
        M1eps_lr_uc.at[i, 'BIC'] = BIC

        M1BIC_all.at[i, 'eps-lr-uc'] = BIC    


        title_excel = os.path.join(save_dir, f'HaS1vol_M1_2modelrecov_eps_lr_uc.xlsx')
        M1eps_lr_uc.to_excel(title_excel, index=False)



            

        #model 5 [eps-lrpos-lrneg]
        #eps, lrpos, lrneg = params
        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_2lr, bounds=bounds5, args=(k,r), strategy=strategy)


        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds5) * np.log(T) + 2*negLL


        #store in dataframe
        M1eps_2lr.at[i, 'eps'] = param_fits[0]
        M1eps_2lr.at[i, 'lrpos'] = param_fits[1]
        M1eps_2lr.at[i, 'lrneg'] = param_fits[2]
        M1eps_2lr.at[i, 'negLL'] = negLL
        M1eps_2lr.at[i, 'BIC'] = BIC

        M1BIC_all.at[i, 'eps-2lr'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1vol_M1_5modelrecov_eps_2lr.xlsx')
        M1eps_2lr.to_excel(title_excel, index=False)




        #model 6 [eps-lrpos-lrneg-uc]
        #eps, lrpos, lrneg, uc = params
        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_2lr_uc, bounds=bounds6, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds6) * np.log(T) + 2*negLL

        #store in dataframe
        M1eps_2lr_uc.at[i, 'eps'] = param_fits[0]
        M1eps_2lr_uc.at[i, 'lrpos'] = param_fits[1]
        M1eps_2lr_uc.at[i, 'lrneg'] = param_fits[2]   
        M1eps_2lr_uc.at[i, 'uc'] = param_fits[3]
        M1eps_2lr_uc.at[i, 'negLL'] = negLL
        M1eps_2lr_uc.at[i, 'BIC'] = BIC
        M1BIC_all.at[i, 'eps-2lr-uc'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1vol_M1_6modelrecov_eps_2lr_uc.xlsx')
        M1eps_2lr_uc.to_excel(title_excel, index=False)


        title_excel = os.path.join(save_dir, f'HaS1vol_M1_modelrecov_BIC_all.xlsx')
        M1BIC_all.to_excel(title_excel, index=False)   

        
    

    #########################################################################################
    #simulate model 2 [eps-lr-uc]
    #########################################################################################
    sample_dir = os.path.join(next_dir2, '2_HaS1vol_modelfit_eps_lr_uc.xlsx')
    sample_pool = pd.read_excel(sample_dir)


    M2BIC_all = pd.DataFrame(index=range(0, 1), columns=['eps-lr', 'eps-lr-uc', 'eps-2lr', 'eps-2lr-uc'])
    M2sampled_eps_lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'uc'])
    M2eps_lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'negLL', 'BIC'])
    M2eps_lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'uc', 'negLL', 'BIC'])
    M2eps_2lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'negLL', 'BIC'])    
    M2eps_2lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'uc', 'negLL', 'BIC'])

    
    for i in range(recov_amount):
        print('M2:', i+1)
        while True:
            ri = np.random.randint(0, len(sample_pool['eps']))
            true_eps = float(sample_pool['eps'].iloc[ri])
            if true_eps < 0.7:
                break
        ri = np.random.randint(0, len(sample_pool['eps']))
        true_lr = float(sample_pool['lr'].iloc[ri])
        ri = np.random.randint(0, len(sample_pool['eps']))
        true_uc = float(sample_pool['uc'].iloc[ri])
        M2sampled_eps_lr_uc.at[i, 'eps'] = true_eps
        M2sampled_eps_lr_uc.at[i, 'lr'] = true_lr
        M2sampled_eps_lr_uc.at[i, 'uc'] = true_uc
        print('true_eps:', true_eps)
        print('true_lr:', true_lr)
        print('true_uc:', true_uc)

        title_excel = os.path.join(save_dir, f'HaS1vol_M2_modelrecov_eps_lr_uc_sampled.xlsx')
        M2sampled_eps_lr_uc.to_excel(title_excel, index=False)

        k, r, Q_k_stored = RW_eps_lr_uc_volatile(eps=true_eps, lr=true_lr, uc= true_uc, T=T, Q_int=Q_int)



        #model 1 [eps-lr]
        #eps, lr = params
        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_lr, bounds=bounds1, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds1) * np.log(T) + 2*negLL

        #store in dataframe
        M2eps_lr.at[i, 'eps'] = param_fits[0]
        M2eps_lr.at[i, 'lr'] = param_fits[1]
        M2eps_lr.at[i, 'negLL'] = negLL
        M2eps_lr.at[i, 'BIC'] = BIC

        M2BIC_all.at[i, 'eps-lr'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1vol_M2_1modelrecov_eps_lr.xlsx')
        M2eps_lr.to_excel(title_excel, index=False)



        #model 2 [eps-lr-uc]
        #eps, lr, uc = params
        negLL = np.inf #initialize negative log likelihood

        result = differential_evolution(negll_RW_eps_lr_uc, bounds=bounds2, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds2) * np.log(T) + 2*negLL

        #store in dataframe
        M2eps_lr_uc.at[i, 'eps'] = param_fits[0]
        M2eps_lr_uc.at[i, 'lr'] = param_fits[1]
        M2eps_lr_uc.at[i, 'uc'] = param_fits[2]
        M2eps_lr_uc.at[i, 'negLL'] = negLL
        M2eps_lr_uc.at[i, 'BIC'] = BIC

        M2BIC_all.at[i, 'eps-lr-uc'] = BIC    


        title_excel = os.path.join(save_dir, f'HaS1vol_M2_2modelrecov_eps_lr_uc.xlsx')
        M2eps_lr_uc.to_excel(title_excel, index=False)



        #model 5 [eps-lrpos-lrneg]
        #eps, lrpos, lrneg = params
        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_2lr, bounds=bounds5, args=(k,r), strategy=strategy)


        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds5) * np.log(T) + 2*negLL


        #store in dataframe
        M2eps_2lr.at[i, 'eps'] = param_fits[0]
        M2eps_2lr.at[i, 'lrpos'] = param_fits[1]
        M2eps_2lr.at[i, 'lrneg'] = param_fits[2]
        M2eps_2lr.at[i, 'negLL'] = negLL
        M2eps_2lr.at[i, 'BIC'] = BIC

        M2BIC_all.at[i, 'eps-2lr'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1vol_M2_5modelrecov_eps_2lr.xlsx')
        M2eps_2lr.to_excel(title_excel, index=False)




        #model 6 [eps-lrpos-lrneg-uc]
        #eps, lrpos, lrneg, uc = params
        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_2lr_uc, bounds=bounds6, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds6) * np.log(T) + 2*negLL

        #store in dataframe
        M2eps_2lr_uc.at[i, 'eps'] = param_fits[0]
        M2eps_2lr_uc.at[i, 'lrpos'] = param_fits[1]
        M2eps_2lr_uc.at[i, 'lrneg'] = param_fits[2]   
        M2eps_2lr_uc.at[i, 'uc'] = param_fits[3]
        M2eps_2lr_uc.at[i, 'negLL'] = negLL
        M2eps_2lr_uc.at[i, 'BIC'] = BIC
        M2BIC_all.at[i, 'eps-2lr-uc'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1vol_M2_6modelrecov_eps_2lr_uc.xlsx')
        M2eps_2lr_uc.to_excel(title_excel, index=False)


        title_excel = os.path.join(save_dir, f'HaS1vol_M2_modelrecov_BIC_all.xlsx')
        M2BIC_all.to_excel(title_excel, index=False)   

    
        

    
    #########################################################################################
    #simulate model 5 [eps-2lr]
    #########################################################################################
    sample_dir = os.path.join(next_dir2, '5_HaS1vol_modelfit_eps_2lr.xlsx')
    sample_pool = pd.read_excel(sample_dir)


    M5BIC_all = pd.DataFrame(index=range(0, 1), columns=['eps-lr', 'eps-lr-uc', 'eps-2lr', 'eps-2lr-uc'])
    M5sampled_eps_2lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos','lrneg'])
    M5eps_lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'negLL', 'BIC'])
    M5eps_lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'uc', 'negLL', 'BIC'])
    M5eps_2lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'negLL', 'BIC'])    
    M5eps_2lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'uc', 'negLL', 'BIC'])


    for i in range(recov_amount):
        print('M5:', i+1)
        while True:
            ri = np.random.randint(0, len(sample_pool['eps']))
            
            # Select parameters
            true_eps = float(sample_pool['eps'].iloc[ri])
            true_lrpos = float(sample_pool['lrpos'].iloc[ri])
            true_lrneg = float(sample_pool['lrneg'].iloc[ri])
            
            # Check both conditions
            if true_eps < 0.7 and abs(true_lrpos - true_lrneg) > 0.3:
                break  # Valid sample found
        M5sampled_eps_2lr.at[i, 'eps'] = true_eps
        M5sampled_eps_2lr.at[i, 'lrpos'] = true_lrpos
        M5sampled_eps_2lr.at[i, 'lrneg'] = true_lrneg
        print('true_eps:', true_eps)
        print('true_lrpos:', true_lrpos)
        print('true_lrneg:', true_lrneg)

        title_excel = os.path.join(save_dir, f'HaS1vol_M5_modelrecov_eps_2lr_sampled.xlsx')
        M5sampled_eps_2lr.to_excel(title_excel, index=False)    

        k, r, Q_k_stored = RW_eps_2lr_volatile(eps=true_eps, lrpos=true_lrpos, lrneg=true_lrneg, T=T, Q_int=Q_int)



        #model 1 [eps-lr]
        #eps, lr = params
        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_lr, bounds=bounds1, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds1) * np.log(T) + 2*negLL

        #store in dataframe
        M5eps_lr.at[i, 'eps'] = param_fits[0]
        M5eps_lr.at[i, 'lr'] = param_fits[1]
        M5eps_lr.at[i, 'negLL'] = negLL
        M5eps_lr.at[i, 'BIC'] = BIC

        M5BIC_all.at[i, 'eps-lr'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1vol_M5_1modelrecov_eps_lr.xlsx')
        M5eps_lr.to_excel(title_excel, index=False)



        #model 2 [eps-lr-uc]
        #eps, lr, uc = params
        negLL = np.inf #initialize negative log likelihood

        result = differential_evolution(negll_RW_eps_lr_uc, bounds=bounds2, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds2) * np.log(T) + 2*negLL

        #store in dataframe
        M5eps_lr_uc.at[i, 'eps'] = param_fits[0]
        M5eps_lr_uc.at[i, 'lr'] = param_fits[1]
        M5eps_lr_uc.at[i, 'uc'] = param_fits[2]
        M5eps_lr_uc.at[i, 'negLL'] = negLL
        M5eps_lr_uc.at[i, 'BIC'] = BIC
        M5BIC_all.at[i, 'eps-lr-uc'] = BIC    


        title_excel = os.path.join(save_dir, f'HaS1vol_M5_2modelrecov_eps_lr_uc.xlsx')
        M5eps_lr_uc.to_excel(title_excel, index=False)



            

        #model 5 [eps-lrpos-lrneg]
        #eps, lrpos, lrneg = params
        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_2lr, bounds=bounds5, args=(k,r), strategy=strategy)


        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds5) * np.log(T) + 2*negLL


        #store in dataframe
        M5eps_2lr.at[i, 'eps'] = param_fits[0]
        M5eps_2lr.at[i, 'lrpos'] = param_fits[1]
        M5eps_2lr.at[i, 'lrneg'] = param_fits[2]
        M5eps_2lr.at[i, 'negLL'] = negLL
        M5eps_2lr.at[i, 'BIC'] = BIC

        M5BIC_all.at[i, 'eps-2lr'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1vol_M5_5modelrecov_eps_2lr.xlsx')
        M5eps_2lr.to_excel(title_excel, index=False)




        #model 6 [eps-lrpos-lrneg-uc]
        #eps, lrpos, lrneg, uc = params
        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_2lr_uc, bounds=bounds6, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds6) * np.log(T) + 2*negLL

        #store in dataframe
        M5eps_2lr_uc.at[i, 'eps'] = param_fits[0]
        M5eps_2lr_uc.at[i, 'lrpos'] = param_fits[1]
        M5eps_2lr_uc.at[i, 'lrneg'] = param_fits[2]   
        M5eps_2lr_uc.at[i, 'uc'] = param_fits[3]
        M5eps_2lr_uc.at[i, 'negLL'] = negLL
        M5eps_2lr_uc.at[i, 'BIC'] = BIC
        M5BIC_all.at[i, 'eps-2lr-uc'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1vol_M5_6modelrecov_eps_2lr_uc.xlsx')
        M5eps_2lr_uc.to_excel(title_excel, index=False)



        title_excel = os.path.join(save_dir, f'HaS1vol_M5_modelrecov_BIC_all.xlsx')
        M5BIC_all.to_excel(title_excel, index=False)   

    
    #########################################################################################
    #simulate model 6 [eps-2lr-uc]
    #########################################################################################
    sample_dir = os.path.join(next_dir2, '6_HaS1vol_modelfit_eps_2lr_uc.xlsx')
    sample_pool = pd.read_excel(sample_dir)


    M6BIC_all = pd.DataFrame(index=range(0, 1), columns=['eps-lr', 'eps-lr-uc', 'eps-2lr', 'eps-2lr-uc'])
    M6sampled_eps_2lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos','lrneg', 'uc'])
    M6eps_lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'negLL', 'BIC'])
    M6eps_lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'uc', 'negLL', 'BIC'])
    M6eps_2lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'negLL', 'BIC'])    
    M6eps_2lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'uc', 'negLL', 'BIC'])

    for i in range(recov_amount):
        print('M6:', i+1)

        while True:
            ri = np.random.randint(0, len(sample_pool['eps']))
            
            # Select parameters
            true_eps = float(sample_pool['eps'].iloc[ri])
            true_lrpos = float(sample_pool['lrpos'].iloc[ri])
            true_lrneg = float(sample_pool['lrneg'].iloc[ri])
            
            # Check both conditions
            if true_eps < 0.7 and abs(true_lrpos - true_lrneg) > 0.3:
                break  # Valid sample found
        ri = np.random.randint(0, len(sample_pool['eps']))
        true_uc = float(sample_pool['uc'].iloc[ri])
        M6sampled_eps_2lr_uc.at[i, 'eps'] = true_eps
        M6sampled_eps_2lr_uc.at[i, 'lrpos'] = true_lrpos
        M6sampled_eps_2lr_uc.at[i, 'lrneg'] = true_lrneg
        M6sampled_eps_2lr_uc.at[i, 'uc'] = true_uc
        print('true_eps:', true_eps)
        print('true_lrpos:', true_lrpos)
        print('true_lrneg:', true_lrneg)
        print('true_uc:', true_uc)

        title_excel = os.path.join(save_dir, f'HaS1sta_M6_modelrecov_eps_2lr_uc_sampled.xlsx')
        M6sampled_eps_2lr_uc.to_excel(title_excel, index=False)

        k, r, Q_k_stored = RW_eps_2lr_uc_volatile(eps=true_eps, lrpos=true_lrpos, lrneg=true_lrneg, uc=true_uc, T=T, Q_int=Q_int)



        #model 1 [eps-lr]
        #eps, lr = params
        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_lr, bounds=bounds1, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds1) * np.log(T) + 2*negLL

        #store in dataframe
        M6eps_lr.at[i, 'eps'] = param_fits[0]
        M6eps_lr.at[i, 'lr'] = param_fits[1]
        M6eps_lr.at[i, 'negLL'] = negLL
        M6eps_lr.at[i, 'BIC'] = BIC

        M6BIC_all.at[i, 'eps-lr'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1vol_M6_1modelrecov_eps_lr.xlsx')
        M6eps_lr.to_excel(title_excel, index=False)



        #model 2 [eps-lr-uc]
        #eps, lr, uc = params
        negLL = np.inf #initialize negative log likelihood

        result = differential_evolution(negll_RW_eps_lr_uc, bounds=bounds2, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds2) * np.log(T) + 2*negLL

        #store in dataframe
        M6eps_lr_uc.at[i, 'eps'] = param_fits[0]
        M6eps_lr_uc.at[i, 'lr'] = param_fits[1]
        M6eps_lr_uc.at[i, 'uc'] = param_fits[2]
        M6eps_lr_uc.at[i, 'negLL'] = negLL
        M6eps_lr_uc.at[i, 'BIC'] = BIC
        M6BIC_all.at[i, 'eps-lr-uc'] = BIC    


        title_excel = os.path.join(save_dir, f'HaS1vol_M6_2modelrecov_eps_lr_uc.xlsx')
        M6eps_lr_uc.to_excel(title_excel, index=False)




        #model 5 [eps-lrpos-lrneg]
        #eps, lrpos, lrneg = params
        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_2lr, bounds=bounds5, args=(k,r), strategy=strategy)


        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds5) * np.log(T) + 2*negLL


        #store in dataframe
        M6eps_2lr.at[i, 'eps'] = param_fits[0]
        M6eps_2lr.at[i, 'lrpos'] = param_fits[1]
        M6eps_2lr.at[i, 'lrneg'] = param_fits[2]
        M6eps_2lr.at[i, 'negLL'] = negLL
        M6eps_2lr.at[i, 'BIC'] = BIC

        M6BIC_all.at[i, 'eps-2lr'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1vol_M6_5modelrecov_eps_2lr.xlsx')
        M6eps_2lr.to_excel(title_excel, index=False)




        #model 6 [eps-lrpos-lrneg-uc]
        #eps, lrpos, lrneg, uc = params
        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_2lr_uc, bounds=bounds6, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds6) * np.log(T) + 2*negLL

        #store in dataframe
        M6eps_2lr_uc.at[i, 'eps'] = param_fits[0]
        M6eps_2lr_uc.at[i, 'lrpos'] = param_fits[1]
        M6eps_2lr_uc.at[i, 'lrneg'] = param_fits[2]   
        M6eps_2lr_uc.at[i, 'uc'] = param_fits[3]
        M6eps_2lr_uc.at[i, 'negLL'] = negLL
        M6eps_2lr_uc.at[i, 'BIC'] = BIC
        M6BIC_all.at[i, 'eps-2lr-uc'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1vol_M6_6modelrecov_eps_2lr_uc.xlsx')
        M6eps_2lr_uc.to_excel(title_excel, index=False)



        title_excel = os.path.join(save_dir, f'HaS1vol_M6_modelrecov_BIC_all.xlsx')
        M6BIC_all.to_excel(title_excel, index=False)   

    


rows = np.arange(recov_amount)
BIC_matrix = np.zeros((4, 4))

M1BIC = M1BIC_all.to_numpy()
M1BICwin = np.zeros_like(M1BIC)
M1_minindices = np.argmin(M1BIC, axis=1)
M1BICwin[rows, M1_minindices] = 1
BIC_matrix[0, :] = np.sum(M1BICwin, axis=0)/recov_amount

M2BIC = M2BIC_all.to_numpy()
M2BICwin = np.zeros_like(M2BIC)
M2_minindices = np.argmin(M2BIC, axis=1)
M2BICwin[rows, M2_minindices] = 1
BIC_matrix[1, :] = np.sum(M2BICwin, axis=0)/recov_amount


M5BIC = M5BIC_all.to_numpy()
M5BICwin = np.zeros_like(M5BIC)
M5_minindices = np.argmin(M5BIC, axis=1)
M5BICwin[rows, M5_minindices] = 1
BIC_matrix[2, :] = np.sum(M5BICwin, axis=0)/recov_amount

M6BIC = M6BIC_all.to_numpy()
M6BICwin = np.zeros_like(M6BIC)
M6_minindices = np.argmin(M6BIC, axis=1)
M6BICwin[rows, M6_minindices] = 1
BIC_matrix[3, :] = np.sum(M6BICwin, axis=0)/recov_amount



simulations = [r'M1: $\epsilon$-$\alpha$', 
               r'M2: $\epsilon$-$\alpha$-$\lambda$',
               r'M5: $\epsilon$-$\alpha_{+}$-$\alpha_{-}$',
               r'M6: $\epsilon$-$\alpha_{+}$-$\alpha_{-}$-$\lambda$',
               ]
model_fits =  [r'M1: $\epsilon$-$\alpha$', 
               r'M2: $\epsilon$-$\alpha$-$\lambda$',
               r'M5: $\epsilon$-$\alpha_{+}$-$\alpha_{-}$',
               r'M6: $\epsilon$-$\alpha_{+}$-$\alpha_{-}$-$\lambda$',
               ]
fig, ax = plt.subplots()
im = ax.imshow(BIC_matrix)

# Show all ticks and label them with the respective list entries
ax.set_xticks(np.arange(len(model_fits)), labels=model_fits, fontsize=15)
ax.set_yticks(np.arange(len(simulations)), labels=simulations, fontsize=15)

# Rotate the tick labels and set their alignment.
plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

# Loop over data dimensions and create text annotations.
for i in range(len(simulations)):
    for j in range(len(model_fits)):
        text = ax.text(j, i, BIC_matrix[i, j],
                       ha="center", va="center", color="w")

ax.set_title("confusion matrix - p(fit model|simulated model)",fontsize=15)
fig.colorbar(im, ax=ax)
plt.show()

