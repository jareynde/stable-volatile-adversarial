#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rescorla Wagner model with epsilon-greedy decision making policy
Constant model
Context:
*Reinforced variability: reward dependent on how variable options are chosen according to Hide and Seek game
    adversarial: least frequent 60% of sequences
            
Model recovery of 8 models in adversarial context,
according to the block structure of HaS experiment 1 (meaning Q-values and freq reset every 200 trials):

*eps-lr
*eps-lr-uc
*eps-lr-lrs
*eps-lr-uc-lrs
*eps-2lr
*eps-2lr-uc
*eps-2lr-lrs
*eps-2lr-uc-lrs


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
def RW_eps_lr_adversarial(eps, lr, T, Q_int):
    #eps        --->        epsilon
    #lr         --->        learning rate
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)

    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.random.uniform(0.9,1.1,K_seq)

    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    
    
    for t in range(T):

        #reset every length of a block of trials in HaS exp 1
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
            seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
            Freq1 = np.random.uniform(0.9,1.1,K_seq)

        
        # store values for Q
        Q_k_stored[t,:] = Q_k   
              
        # make choice based on choice probababilities
        rand[t] = np.random.choice([0,1], p=[1-eps,eps])
        if rand[t] == 0:
            k[t] = np.argmax(Q_k)
        if rand[t] == 1:
            k[t] = np.random.choice(range(K))
        
        #variable
        if t < 1:
            r[t] = random.choice([0, 1])
        else: 
            current_seq = k[t-1:t+1]
            current_index = np.where(np.all(seq_options1==current_seq,axis=1))[0]
            current_freq = Freq1[current_index]
            if current_freq < np.percentile(Freq1,60):
                r[t] = 1
            else: r[t] = 0

            Adding = np.ones(64, dtype=float)*(-1/63)
            Freq1 = np.add(Freq1, Adding)
            Freq1[current_index] = Freq1[current_index] + 1 + (1/63)
            Freq1 = Freq1*0.984

         # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + lr * delta_k


    return k, r, Q_k_stored


#Model 2:
def RW_eps_lr_uc_adversarial(eps, lr, uc, T, Q_int):
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

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.random.uniform(0.9,1.1,K_seq)

    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    
    
    for t in range(T):

        #reset every length of a block of trials in HaS exp 1
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
            seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
            Freq1 = np.random.uniform(0.9,1.1,K_seq)
        
        # store values for Q
        Q_k_stored[t,:] = Q_k   
              
        # make choice based on choice probababilities
        rand[t] = np.random.choice([0,1], p=[1-eps,eps])
        if rand[t] == 0:
            k[t] = np.argmax(Q_k)
        if rand[t] == 1:
            k[t] = np.random.choice(range(K))
        
        #variable
        if t < 1:
            r[t] = random.choice([0, 1])
        else: 
            current_seq = k[t-1:t+1]
            current_index = np.where(np.all(seq_options1==current_seq,axis=1))[0]
            current_freq = Freq1[current_index]
            if current_freq < np.percentile(Freq1,60):
                r[t] = 1
            else: r[t] = 0

            Adding = np.ones(64, dtype=float)*(-1/63)
            Freq1 = np.add(Freq1, Adding)
            Freq1[current_index] = Freq1[current_index] + 1 + (1/63)
            Freq1 = Freq1*0.984

         # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + lr * delta_k

        # update Q values for unchosen option:
        Q_k += uc  # Apply the unchosen bias to all Q-values
        Q_k[k[t]] -= uc  # Counteract the bias for the chosen option to keep it balanced

    return k, r, Q_k_stored

#Model 3:
def RW_eps_lr_lrs_adversarial(eps, lr, lrs, w, T, Q_int):
    #eps        --->        epsilon
    #lr         --->        learning rate
    #lrs        --->        learning rate for sequences
    #w          --->        weight for learned values (Q values and Q sequence values)
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)

    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.random.uniform(0.9,1.1,K_seq)

    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    
    S_k_stored = np.zeros((T,K_seq), dtype=float)
    S_k = np.ones(K_seq)*Q_int

    for t in range(T):

        #reset every length of a block of trials in HaS exp 1
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
            seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
            Freq1 = np.random.uniform(0.9,1.1,K_seq)
            S_k = np.ones(K_seq)*Q_int #initual value of Q for each choice
       
        # store values for Q
        Q_k_stored[t,:] = Q_k  
        S_k_stored[t,:] = S_k 
            
     
        if t == 0:
            next_seq_options = np.zeros(K)
    
        # make choice based on choice probababilities
        combined_QandS_info = w*Q_k + (1-w)*next_seq_options
        rand[t] = np.random.choice([0,1], p=[1-eps,eps])
        if rand[t] == 0:
            k[t] = np.argmax(combined_QandS_info)
        if rand[t] == 1:
            k[t] = np.random.choice(range(K))
        
        #variable
        if t < 1:
            r[t] = random.choice([0, 1])
        else: 
            current_seq = k[t-1:t+1]
            current_index = np.where(np.all(seq_options1==current_seq,axis=1))[0]
            current_freq = Freq1[current_index]
            if current_freq < np.percentile(Freq1,60):
                r[t] = 1
            else: r[t] = 0

            Adding = np.ones(64, dtype=float)*(-1/63)
            Freq1 = np.add(Freq1, Adding)
            Freq1[current_index] = Freq1[current_index] + 1 + (1/63)
            Freq1 = Freq1*0.984

            # update Q_values of sequences
            sdelta_k = r[t] - S_k[current_index]
            S_k[current_index] = S_k[current_index] + lrs * sdelta_k

            #if the current choice is x, next_seq_options stores the S values of all possible next response pairs (that start with x)
            next_seq_options = np.zeros(K)
            for i in range(K):
                next_seq = [k[t], i]
                next_index = np.where(np.all(seq_options1==next_seq,axis=1))[0][0]
                next_seq_options[i] = S_k[next_index]

         # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + lr * delta_k


    return k, r, Q_k_stored


#Model 4:
def RW_eps_lr_uc_lrs_adversarial(eps, lr, uc, lrs, w, T, Q_int):
    #eps        --->        epsilon
    #lr         --->        learning rate
    #uc         --->        unchosen value-bias
    #lrs        --->        learning rate for sequences
    #w          --->        weight for learned values (Q values and Q sequence values)
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)

    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.random.uniform(0.9,1.1,K_seq)

    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    
    S_k_stored = np.zeros((T,K_seq), dtype=float)
    S_k = np.ones(K_seq)*Q_int

    for t in range(T):

        #reset every length of a block of trials in HaS exp 1
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
            seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
            Freq1 = np.random.uniform(0.9,1.1,K_seq)
            S_k = np.ones(K_seq)*Q_int #initual value of Q for each choice
        
        # store values for Q
        Q_k_stored[t,:] = Q_k  
        S_k_stored[t,:] = S_k 
            
     
        if t == 0:
            next_seq_options = np.zeros(K)
    
        # make choice based on choice probababilities
        combined_QandS_info = w*Q_k + (1-w)*next_seq_options
        rand[t] = np.random.choice([0,1], p=[1-eps,eps])
        if rand[t] == 0:
            k[t] = np.argmax(combined_QandS_info)
        if rand[t] == 1:
            k[t] = np.random.choice(range(K))
        
        #variable
        if t < 1:
            r[t] = random.choice([0, 1])
        else: 
            current_seq = k[t-1:t+1]
            current_index = np.where(np.all(seq_options1==current_seq,axis=1))[0]
            current_freq = Freq1[current_index]
            if current_freq < np.percentile(Freq1,60):
                r[t] = 1
            else: r[t] = 0

            Adding = np.ones(64, dtype=float)*(-1/63)
            Freq1 = np.add(Freq1, Adding)
            Freq1[current_index] = Freq1[current_index] + 1 + (1/63)
            Freq1 = Freq1*0.984

            # update Q_values of sequences
            sdelta_k = r[t] - S_k[current_index]
            S_k[current_index] = S_k[current_index] + lrs * sdelta_k

            #if the current choice is x, next_seq_options stores the S values of all possible next response pairs (that start with x)
            next_seq_options = np.zeros(K)
            for i in range(K):
                next_seq = [k[t], i]
                next_index = np.where(np.all(seq_options1==next_seq,axis=1))[0][0]
                next_seq_options[i] = S_k[next_index]

        # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        Q_k[k[t]] = Q_k[k[t]] + lr * delta_k

        # update Q values for unchosen option:
        Q_k += uc  # Apply the unchosen bias to all Q-values
        Q_k[k[t]] -= uc  # Counteract the bias for the chosen option to keep it balanced


    return k, r, Q_k_stored

#Model 5:
def RW_eps_2lr_adversarial(eps, lrpos, lrneg, T, Q_int):
    #eps        --->        epsilon
    #lr         --->        learning rate
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)

    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.random.uniform(0.9,1.1,K_seq)

    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    
    
    for t in range(T):

        #reset every length of a block of trials in HaS exp 1
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
            seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
            Freq1 = np.random.uniform(0.9,1.1,K_seq)
        
        # store values for Q
        Q_k_stored[t,:] = Q_k   
              
        # make choice based on choice probababilities
        rand[t] = np.random.choice([0,1], p=[1-eps,eps])
        if rand[t] == 0:
            k[t] = np.argmax(Q_k)
        if rand[t] == 1:
            k[t] = np.random.choice(range(K))
        
        #variable
        if t < 1:
            r[t] = random.choice([0, 1])
        else: 
            current_seq = k[t-1:t+1]
            current_index = np.where(np.all(seq_options1==current_seq,axis=1))[0]
            current_freq = Freq1[current_index]
            if current_freq < np.percentile(Freq1,60):
                r[t] = 1
            else: r[t] = 0

            Adding = np.ones(64, dtype=float)*(-1/63)
            Freq1 = np.add(Freq1, Adding)
            Freq1[current_index] = Freq1[current_index] + 1 + (1/63)
            Freq1 = Freq1*0.984

         # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        if delta_k < 0: 
            Q_k[k[t]] = Q_k[k[t]] + lrneg * delta_k
        if delta_k >= 0:
            Q_k[k[t]] = Q_k[k[t]] + lrpos * delta_k


    return k, r, Q_k_stored


#Model 6:
def RW_eps_2lr_uc_adversarial(eps, lrpos, lrneg, uc, T, Q_int):
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

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.random.uniform(0.9,1.1,K_seq)

    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    
    
    for t in range(T):

        #reset every length of a block of trials in HaS exp 1
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
            seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
            Freq1 = np.random.uniform(0.9,1.1,K_seq)
        
        # store values for Q
        Q_k_stored[t,:] = Q_k   
              
        # make choice based on choice probababilities
        rand[t] = np.random.choice([0,1], p=[1-eps,eps])
        if rand[t] == 0:
            k[t] = np.argmax(Q_k)
        if rand[t] == 1:
            k[t] = np.random.choice(range(K))
        
        #variable
        if t < 1:
            r[t] = random.choice([0, 1])
        else: 
            current_seq = k[t-1:t+1]
            current_index = np.where(np.all(seq_options1==current_seq,axis=1))[0]
            current_freq = Freq1[current_index]
            if current_freq < np.percentile(Freq1,60):
                r[t] = 1
            else: r[t] = 0

            Adding = np.ones(64, dtype=float)*(-1/63)
            Freq1 = np.add(Freq1, Adding)
            Freq1[current_index] = Freq1[current_index] + 1 + (1/63)
            Freq1 = Freq1*0.984

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


#Model 7:
def RW_eps_2lr_lrs_adversarial(eps, lrpos, lrneg, lrs, w, T, Q_int):
    #eps        --->        epsilon
    #lr         --->        learning rate
    #lrs        --->        learning rate for sequences
    #w          --->        weight for learned values (Q values and Q sequence values)
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)

    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.random.uniform(0.9,1.1,K_seq)

    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    
    S_k_stored = np.zeros((T,K_seq), dtype=float)
    S_k = np.ones(K_seq)*Q_int
    
    for t in range(T):

        #reset every length of a block of trials in HaS exp 1
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
            seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
            Freq1 = np.random.uniform(0.9,1.1,K_seq)
            S_k = np.ones(K_seq)*Q_int #initual value of Q for each choice
        
        # store values for Q
        Q_k_stored[t,:] = Q_k   
        S_k_stored[t,:] = S_k 

        if t == 0:
            next_seq_options = np.zeros(K)
              
        # make choice based on choice probababilities
        combined_QandS_info = w*Q_k + (1-w)*next_seq_options
        rand[t] = np.random.choice([0,1], p=[1-eps,eps])
        if rand[t] == 0:
            k[t] = np.argmax(combined_QandS_info)
        if rand[t] == 1:
            k[t] = np.random.choice(range(K))
        
        #variable
        if t < 1:
            r[t] = random.choice([0, 1])
        else: 
            current_seq = k[t-1:t+1]
            current_index = np.where(np.all(seq_options1==current_seq,axis=1))[0]
            current_freq = Freq1[current_index]
            if current_freq < np.percentile(Freq1,60):
                r[t] = 1
            else: r[t] = 0

            Adding = np.ones(64, dtype=float)*(-1/63)
            Freq1 = np.add(Freq1, Adding)
            Freq1[current_index] = Freq1[current_index] + 1 + (1/63)
            Freq1 = Freq1*0.984

            # update Q_values of sequences
            sdelta_k = r[t] - S_k[current_index]
            S_k[current_index] = S_k[current_index] + lrs * sdelta_k

            #if the current choice is x, next_seq_options stores the S values of all possible next response pairs (that start with x)
            next_seq_options = np.zeros(K)
            for i in range(K):
                next_seq = [k[t], i]
                next_index = np.where(np.all(seq_options1==next_seq,axis=1))[0][0]
                next_seq_options[i] = S_k[next_index]

         # update Q values for chosen option:
        delta_k = r[t] - Q_k[k[t]]
        if delta_k < 0: 
            Q_k[k[t]] = Q_k[k[t]] + lrneg * delta_k
        if delta_k >= 0:
            Q_k[k[t]] = Q_k[k[t]] + lrpos * delta_k

    return k, r, Q_k_stored



#Model 8:
def RW_eps_2lr_uc_lrs_adversarial(eps, lrpos, lrneg, uc, lrs, w, T, Q_int):
    #eps        --->        epsilon
    #lrpos      --->        positive PE learning rate
    #lrneg      --->        negative PE learning rate
    #uc         --->        unchosen value-bias    
    #lrs        --->        learning rate for sequences
    #w          --->        weight for learned values (Q values and Q sequence values)
    #T          --->        amount of trials for each simulation
    #Q_int      --->        parameter that determines initial Q value: intial Q values for all actions are equal to Q_int*(1/K)

    K=8 #the amount of choice options
    K_seq = 64 #the amount of choice sequences (1 sequence consists of 6 consecutive binary choices)
    
    k = np.zeros((T), dtype=int) #vector of choices made
    rand = np.zeros((T), dtype=int) #vector of choices made
    r = np.zeros((T), dtype=float) #vector of rewards

    #for adversarial env
    seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
    Freq1 = np.random.uniform(0.9,1.1,K_seq)

    #Q values
    Q_k_stored = np.zeros((T,K), dtype = float) #Q value vector for each choice for each action
    Q_k = np.ones(K)*Q_int #initual value of Q for each choice
    
    S_k_stored = np.zeros((T,K_seq), dtype=float)
    S_k = np.ones(K_seq)*Q_int
    
    for t in range(T):

        #reset every length of a block of trials in HaS exp 1
        B = 200 #number of trials in one block of experiment
        if (t%B) == 0:
            Q_k = np.ones(K)*Q_int #initual value of Q for each choice
            seq_options1 = np.array([[a, b] for a in range(8) for b in range(8)])
            Freq1 = np.random.uniform(0.9,1.1,K_seq)
            S_k = np.ones(K_seq)*Q_int #initual value of Q for each choice
        
        # store values for Q
        Q_k_stored[t,:] = Q_k   
        S_k_stored[t,:] = S_k 

        if t == 0:
            next_seq_options = np.zeros(K)
              
        # make choice based on choice probababilities
        combined_QandS_info = w*Q_k + (1-w)*next_seq_options
        rand[t] = np.random.choice([0,1], p=[1-eps,eps])
        if rand[t] == 0:
            k[t] = np.argmax(combined_QandS_info)
        if rand[t] == 1:
            k[t] = np.random.choice(range(K))
        
        #variable
        if t < 1:
            r[t] = random.choice([0, 1])
        else: 
            current_seq = k[t-1:t+1]
            current_index = np.where(np.all(seq_options1==current_seq,axis=1))[0]
            current_freq = Freq1[current_index]
            if current_freq < np.percentile(Freq1,60):
                r[t] = 1
            else: r[t] = 0

            Adding = np.ones(64, dtype=float)*(-1/63)
            Freq1 = np.add(Freq1, Adding)
            Freq1[current_index] = Freq1[current_index] + 1 + (1/63)
            Freq1 = Freq1*0.984

            # update Q_values of sequences
            sdelta_k = r[t] - S_k[current_index]
            S_k[current_index] = S_k[current_index] + lrs * sdelta_k

            #if the current choice is x, next_seq_options stores the S values of all possible next response pairs (that start with x)
            next_seq_options = np.zeros(K)
            for i in range(K):
                next_seq = [k[t], i]
                next_index = np.where(np.all(seq_options1==next_seq,axis=1))[0][0]
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
recov_amount = 100
Q_int = 1

bounds1=[(0,1), (0,1)]
bounds2=[(0,1), (0,1),(-3,3)]
bounds3=[(0,1), (0,1), (0,1), (0,1)]
bounds4=[(0,1), (0,1), (-3,3), (0,1), (0,1)]

bounds5=[(0,1), (0,1), (0,1)]
bounds6=[(0,1), (0,1), (0,1),(-3,3)]
bounds7=[(0,1), (0,1), (0,1), (0,1), (0,1)]
bounds8=[(0,1), (0,1), (0,1), (-3,3), (0,1), (0,1)]



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

    '''
    #########################################################################################
    #simulate model 1 [eps-lr]
    #########################################################################################
    sample_dir = os.path.join(next_dir2, '1_HaS1var_modelfit_eps_lr.xlsx')
    sample_pool = pd.read_excel(sample_dir)
    print(sample_pool)

    M1BIC_all = pd.DataFrame(index=range(0, 1), columns=['eps-lr', 'eps-lr-uc', 'eps-lr-lrs', 'eps-lr-uc-lrs', 'eps-2lr', 'eps-2lr-uc', 'eps-2lr-lrs', 'eps-2lr-uc-lrs'])
    M1sampled_eps_lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr'])
    M1eps_lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'negLL', 'BIC'])
    M1eps_lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'uc', 'negLL', 'BIC'])
    M1eps_lr_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'lrs', 'w', 'negLL', 'BIC'])
    M1eps_lr_uc_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'uc', 'lrs', 'w', 'negLL', 'BIC'])    
    M1eps_2lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'negLL', 'BIC'])    
    M1eps_2lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'uc', 'negLL', 'BIC'])
    M1eps_2lr_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'lrs', 'w', 'negLL', 'BIC'])
    M1eps_2lr_uc_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'uc', 'lrs', 'w', 'negLL', 'BIC'])

    for i in range(recov_amount):
        print(sample_pool['eps'])
        ri = np.random.randint(0, len(sample_pool['eps']))
        # parameter selection from estimated values from has data
        true_eps = float(sample_pool['eps'].iloc[ri])
        ri = np.random.randint(0, len(sample_pool['eps']))
        true_lr = float(sample_pool['lr'].iloc[ri])
        M1sampled_eps_lr.at[i, 'eps'] = true_eps
        M1sampled_eps_lr.at[i, 'lr'] = true_lr

        title_excel = os.path.join(save_dir, f'HaS1var_M1_modelrecov_eps_lr_sampled.xlsx')
        M1sampled_eps_lr.to_excel(title_excel, index=False)

        k, r, Q_k_stored = RW_eps_lr_adversarial(eps=true_eps, lr=true_lr, T=T, Q_int=Q_int)



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

        title_excel = os.path.join(save_dir, f'HaS1var_M1_1modelrecov_eps_lr.xlsx')
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


        title_excel = os.path.join(save_dir, f'HaS1var_M1_2modelrecov_eps_lr_uc.xlsx')
        M1eps_lr_uc.to_excel(title_excel, index=False)



        #model 3 [eps-lr-lrs]
        #eps, lr, lrs, w = params
        negLL = np.inf #initialize negative log likelihood

        result = differential_evolution(negll_RW_eps_lr_lrs, bounds=bounds3, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds3) * np.log(T) + 2*negLL

        #store in dataframe
        M1eps_lr_lrs.at[i, 'eps'] = param_fits[0]
        M1eps_lr_lrs.at[i, 'lr'] = param_fits[1]
        M1eps_lr_lrs.at[i, 'lrs'] = param_fits[2]
        M1eps_lr_lrs.at[i, 'w'] = param_fits[3]
        M1eps_lr_lrs.at[i, 'negLL'] = negLL
        M1eps_lr_lrs.at[i, 'BIC'] = BIC

        M1BIC_all.at[i, 'eps-lr-lrs'] = BIC
        title_excel = os.path.join(save_dir, f'HaS1var_M1_3modelrecov_eps_lr_lrs.xlsx')
        M1eps_lr_lrs.to_excel(title_excel, index=False)

            


        #model 4 [eps-lr-uc-lrs]
        #eps, lr, uc, lrs, w = params

        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_lr_uc_lrs, bounds=bounds4, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds4) * np.log(T) + 2*negLL

        M1eps_lr_uc_lrs.at[i, 'eps'] = param_fits[0]
        M1eps_lr_uc_lrs.at[i, 'lr'] = param_fits[1]
        M1eps_lr_uc_lrs.at[i, 'uc'] = param_fits[2]
        M1eps_lr_uc_lrs.at[i, 'lrs'] = param_fits[3]
        M1eps_lr_uc_lrs.at[i, 'w'] = param_fits[4]
        M1eps_lr_uc_lrs.at[i, 'negLL'] = negLL
        M1eps_lr_uc_lrs.at[i, 'BIC'] = BIC

        M1BIC_all.at[i, 'eps-lr-uc-lrs'] = BIC
        title_excel = os.path.join(save_dir, f'HaS1var_M1_4modelrecov_eps_lr_uc_lrs.xlsx')
        M1eps_lr_uc_lrs.to_excel(title_excel, index=False)


            

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

        title_excel = os.path.join(save_dir, f'HaS1var_M1_5modelrecov_eps_2lr.xlsx')
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

        title_excel = os.path.join(save_dir, f'HaS1var_M1_6modelrecov_eps_2lr_uc.xlsx')
        M1eps_2lr_uc.to_excel(title_excel, index=False)





        #model 7 [eps-lrpos-lrneg-lrs]
        #eps, lrpos, lrneg, lrs, w = params

        negLL = np.inf #initialize negative log likelihood

        result = differential_evolution(negll_RW_eps_2lr_lrs, bounds=bounds7, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds7) * np.log(T) + 2*negLL

        #store in dataframe
        M1eps_2lr_lrs.at[i, 'eps'] = param_fits[0]
        M1eps_2lr_lrs.at[i, 'lrpos'] = param_fits[1]
        M1eps_2lr_lrs.at[i, 'lrneg'] = param_fits[2]
        M1eps_2lr_lrs.at[i, 'lrs'] = param_fits[3]
        M1eps_2lr_lrs.at[i, 'w'] = param_fits[4]
        M1eps_2lr_lrs.at[i, 'negLL'] = negLL
        M1eps_2lr_lrs.at[i, 'BIC'] = BIC
        M1BIC_all.at[i, 'eps-2lr-lrs'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1var_M1_7modelrecov_eps_2lr_lrs.xlsx')
        M1eps_2lr_lrs.to_excel(title_excel, index=False)




        #model 8 [eps-lrpos-lrneg-uc-lrs]
        #eps, lrpos, lrneg, uc, lrs, w = params
        negLL = np.inf #initialize negative log likelihood

        #eps, lrpos, lrneg, lrs, w = params
        result = differential_evolution(negll_RW_eps_2lr_uc_lrs, bounds=bounds8, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds8) * np.log(T) + 2*negLL

        #store in dataframe
        M1eps_2lr_uc_lrs.at[i, 'eps'] = param_fits[0]
        M1eps_2lr_uc_lrs.at[i, 'lrpos'] = param_fits[1]
        M1eps_2lr_uc_lrs.at[i, 'lrneg'] = param_fits[2]
        M1eps_2lr_uc_lrs.at[i, 'uc'] = param_fits[3]
        M1eps_2lr_uc_lrs.at[i, 'lrs'] = param_fits[4]
        M1eps_2lr_uc_lrs.at[i, 'w'] = param_fits[5]
        M1eps_2lr_uc_lrs.at[i, 'negLL'] = negLL
        M1eps_2lr_uc_lrs.at[i, 'BIC'] = BIC
        M1BIC_all.at[i, 'eps-2lr-uc-lrs'] = BIC
        title_excel = os.path.join(save_dir, f'HaS1var_M1_8modelrecov_eps_2lr_uc_lrs.xlsx')
        M1eps_2lr_uc_lrs.to_excel(title_excel, index=False)

        title_excel = os.path.join(save_dir, f'HaS1var_M1_modelrecov_BIC_all.xlsx')
        M1BIC_all.to_excel(title_excel, index=False)   

        
    

    #########################################################################################
    #simulate model 2 [eps-lr-uc]
    #########################################################################################
    sample_dir = os.path.join(next_dir2, '2_HaS1var_modelfit_eps_lr_uc.xlsx')
    sample_pool = pd.read_excel(sample_dir)


    M2BIC_all = pd.DataFrame(index=range(0, 1), columns=['eps-lr', 'eps-lr-uc', 'eps-lr-lrs', 'eps-lr-uc-lrs', 'eps-2lr', 'eps-2lr-uc', 'eps-2lr-lrs', 'eps-2lr-uc-lrs'])
    M2sampled_eps_lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'uc'])
    M2eps_lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'negLL', 'BIC'])
    M2eps_lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'uc', 'negLL', 'BIC'])
    M2eps_lr_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'lrs', 'w', 'negLL', 'BIC'])
    M2eps_lr_uc_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'uc', 'lrs', 'w', 'negLL', 'BIC'])    
    M2eps_2lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'negLL', 'BIC'])    
    M2eps_2lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'uc', 'negLL', 'BIC'])
    M2eps_2lr_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'lrs', 'w', 'negLL', 'BIC'])
    M2eps_2lr_uc_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'uc', 'lrs', 'w', 'negLL', 'BIC'])

    
    for i in range(recov_amount):
        print('M2:', i+1)
        ri = np.random.randint(0, len(sample_pool['eps']))
        # parameter selection from estimated values from has data
        true_eps = float(sample_pool['eps'].iloc[ri])
        ri = np.random.randint(0, len(sample_pool['eps']))
        true_lr = float(sample_pool['lr'].iloc[ri])
        ri = np.random.randint(0, len(sample_pool['eps']))
        true_uc = float(sample_pool['uc'].iloc[ri])
        M2sampled_eps_lr_uc.at[i, 'eps'] = true_eps
        M2sampled_eps_lr_uc.at[i, 'lr'] = true_lr
        M2sampled_eps_lr_uc.at[i, 'uc'] = true_uc

        title_excel = os.path.join(save_dir, f'HaS1var_M2_modelrecov_eps_lr_uc_sampled.xlsx')
        M2sampled_eps_lr_uc.to_excel(title_excel, index=False)

        k, r, Q_k_stored = RW_eps_lr_uc_adversarial(eps=true_eps, lr=true_lr, uc= true_uc, T=T, Q_int=Q_int)



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

        title_excel = os.path.join(save_dir, f'HaS1var_M2_1modelrecov_eps_lr.xlsx')
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


        title_excel = os.path.join(save_dir, f'HaS1var_M2_2modelrecov_eps_lr_uc.xlsx')
        M2eps_lr_uc.to_excel(title_excel, index=False)



        #model 3 [eps-lr-lrs]
        #eps, lr, lrs, w = params
        negLL = np.inf #initialize negative log likelihood

        result = differential_evolution(negll_RW_eps_lr_lrs, bounds=bounds3, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds3) * np.log(T) + 2*negLL

        #store in dataframe
        M2eps_lr_lrs.at[i, 'eps'] = param_fits[0]
        M2eps_lr_lrs.at[i, 'lr'] = param_fits[1]
        M2eps_lr_lrs.at[i, 'lrs'] = param_fits[2]
        M2eps_lr_lrs.at[i, 'w'] = param_fits[3]
        M2eps_lr_lrs.at[i, 'negLL'] = negLL
        M2eps_lr_lrs.at[i, 'BIC'] = BIC

        M2BIC_all.at[i, 'eps-lr-lrs'] = BIC
        title_excel = os.path.join(save_dir, f'HaS1var_M2_3modelrecov_eps_lr_lrs.xlsx')
        M2eps_lr_lrs.to_excel(title_excel, index=False)

            


        #model 4 [eps-lr-uc-lrs]
        #eps, lr, uc, lrs, w = params

        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_lr_uc_lrs, bounds=bounds4, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds4) * np.log(T) + 2*negLL

        M2eps_lr_uc_lrs.at[i, 'eps'] = param_fits[0]
        M2eps_lr_uc_lrs.at[i, 'lr'] = param_fits[1]
        M2eps_lr_uc_lrs.at[i, 'uc'] = param_fits[2]
        M2eps_lr_uc_lrs.at[i, 'lrs'] = param_fits[3]
        M2eps_lr_uc_lrs.at[i, 'w'] = param_fits[4]
        M2eps_lr_uc_lrs.at[i, 'negLL'] = negLL
        M2eps_lr_uc_lrs.at[i, 'BIC'] = BIC

        M2BIC_all.at[i, 'eps-lr-uc-lrs'] = BIC
        title_excel = os.path.join(save_dir, f'HaS1var_M2_4modelrecov_eps_lr_uc_lrs.xlsx')
        M2eps_lr_uc_lrs.to_excel(title_excel, index=False)


            

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

        title_excel = os.path.join(save_dir, f'HaS1var_M2_5modelrecov_eps_2lr.xlsx')
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

        title_excel = os.path.join(save_dir, f'HaS1var_M2_6modelrecov_eps_2lr_uc.xlsx')
        M2eps_2lr_uc.to_excel(title_excel, index=False)





        #model 7 [eps-lrpos-lrneg-lrs]
        #eps, lrpos, lrneg, lrs, w = params

        negLL = np.inf #initialize negative log likelihood

        result = differential_evolution(negll_RW_eps_2lr_lrs, bounds=bounds7, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds7) * np.log(T) + 2*negLL

        #store in dataframe
        M2eps_2lr_lrs.at[i, 'eps'] = param_fits[0]
        M2eps_2lr_lrs.at[i, 'lrpos'] = param_fits[1]
        M2eps_2lr_lrs.at[i, 'lrneg'] = param_fits[2]
        M2eps_2lr_lrs.at[i, 'lrs'] = param_fits[3]
        M2eps_2lr_lrs.at[i, 'w'] = param_fits[4]
        M2eps_2lr_lrs.at[i, 'negLL'] = negLL
        M2eps_2lr_lrs.at[i, 'BIC'] = BIC
        M2BIC_all.at[i, 'eps-2lr-lrs'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1var_M2_7modelrecov_eps_2lr_lrs.xlsx')
        M2eps_2lr_lrs.to_excel(title_excel, index=False)




        #model 8 [eps-lrpos-lrneg-uc-lrs]
        #eps, lrpos, lrneg, uc, lrs, w = params
        negLL = np.inf #initialize negative log likelihood

        #eps, lrpos, lrneg, lrs, w = params
        result = differential_evolution(negll_RW_eps_2lr_uc_lrs, bounds=bounds8, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds8) * np.log(T) + 2*negLL

        #store in dataframe
        M2eps_2lr_uc_lrs.at[i, 'eps'] = param_fits[0]
        M2eps_2lr_uc_lrs.at[i, 'lrpos'] = param_fits[1]
        M2eps_2lr_uc_lrs.at[i, 'lrneg'] = param_fits[2]
        M2eps_2lr_uc_lrs.at[i, 'uc'] = param_fits[3]
        M2eps_2lr_uc_lrs.at[i, 'lrs'] = param_fits[4]
        M2eps_2lr_uc_lrs.at[i, 'w'] = param_fits[5]
        M2eps_2lr_uc_lrs.at[i, 'negLL'] = negLL
        M2eps_2lr_uc_lrs.at[i, 'BIC'] = BIC
        M2BIC_all.at[i, 'eps-2lr-uc-lrs'] = BIC
        title_excel = os.path.join(save_dir, f'HaS1var_M2_8modelrecov_eps_2lr_uc_lrs.xlsx')
        M2eps_2lr_uc_lrs.to_excel(title_excel, index=False)

        title_excel = os.path.join(save_dir, f'HaS1var_M2_modelrecov_BIC_all.xlsx')
        M2BIC_all.to_excel(title_excel, index=False)   

        


    

    #########################################################################################
    #simulate model 3 [eps-lr-lrs]
    #########################################################################################
    sample_dir = os.path.join(next_dir2, '3_HaS1var_modelfit_eps_lr_lrs.xlsx')
    sample_pool = pd.read_excel(sample_dir)


    M3BIC_all = pd.DataFrame(index=range(0, 1), columns=['eps-lr', 'eps-lr-uc', 'eps-lr-lrs', 'eps-lr-uc-lrs', 'eps-2lr', 'eps-2lr-uc', 'eps-2lr-lrs', 'eps-2lr-uc-lrs'])
    M3sampled_eps_lr_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'lrs', 'w'])
    M3eps_lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'negLL', 'BIC'])
    M3eps_lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'uc', 'negLL', 'BIC'])
    M3eps_lr_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'lrs', 'w', 'negLL', 'BIC'])
    M3eps_lr_uc_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'uc', 'lrs', 'w', 'negLL', 'BIC'])    
    M3eps_2lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'negLL', 'BIC'])    
    M3eps_2lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'uc', 'negLL', 'BIC'])
    M3eps_2lr_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'lrs', 'w', 'negLL', 'BIC'])
    M3eps_2lr_uc_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'uc', 'lrs', 'w', 'negLL', 'BIC'])

    for i in range(recov_amount):
        print('M3:', i+1)
        ri = np.random.randint(0, len(sample_pool['eps']))
        # parameter selection from estimated values from has data
        true_eps = float(sample_pool['eps'].iloc[ri])
        ri = np.random.randint(0, len(sample_pool['eps']))
        true_lr = float(sample_pool['lr'].iloc[ri])
        ri = np.random.randint(0, len(sample_pool['eps']))
        true_lrs = float(sample_pool['lrs'].iloc[ri])
        ri = np.random.randint(0, len(sample_pool['eps']))
        true_w = float(sample_pool['w'].iloc[ri])
        M3sampled_eps_lr_lrs.at[i, 'eps'] = true_eps
        M3sampled_eps_lr_lrs.at[i, 'lr'] = true_lr
        M3sampled_eps_lr_lrs.at[i, 'lrs'] = true_lrs
        M3sampled_eps_lr_lrs.at[i, 'w'] = true_w
        title_excel = os.path.join(save_dir, f'HaS1var_M3_modelrecov_eps_lr_lrs_sampled.xlsx')
        M3sampled_eps_lr_lrs.to_excel(title_excel, index=False)

        k, r, Q_k_stored = RW_eps_lr_lrs_adversarial(eps=true_eps, lr=true_lr, lrs= true_lrs, w=true_w, T=T, Q_int=Q_int)



        #model 1 [eps-lr]
        #eps, lr = params
        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_lr, bounds=bounds1, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds1) * np.log(T) + 2*negLL

        #store in dataframe
        M3eps_lr.at[i, 'eps'] = param_fits[0]
        M3eps_lr.at[i, 'lr'] = param_fits[1]
        M3eps_lr.at[i, 'negLL'] = negLL
        M3eps_lr.at[i, 'BIC'] = BIC

        M3BIC_all.at[i, 'eps-lr'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1var_M3_1modelrecov_eps_lr.xlsx')
        M3eps_lr.to_excel(title_excel, index=False)



        #model 2 [eps-lr-uc]
        #eps, lr, uc = params
        negLL = np.inf #initialize negative log likelihood

        result = differential_evolution(negll_RW_eps_lr_uc, bounds=bounds2, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds2) * np.log(T) + 2*negLL

        #store in dataframe
        M3eps_lr_uc.at[i, 'eps'] = param_fits[0]
        M3eps_lr_uc.at[i, 'lr'] = param_fits[1]
        M3eps_lr_uc.at[i, 'uc'] = param_fits[2]
        M3eps_lr_uc.at[i, 'negLL'] = negLL
        M3eps_lr_uc.at[i, 'BIC'] = BIC
        M3BIC_all.at[i, 'eps-lr-uc'] = BIC    


        title_excel = os.path.join(save_dir, f'HaS1var_M3_2modelrecov_eps_lr_uc.xlsx')
        M3eps_lr_uc.to_excel(title_excel, index=False)



        #model 3 [eps-lr-lrs]
        #eps, lr, lrs, w = params
        negLL = np.inf #initialize negative log likelihood

        result = differential_evolution(negll_RW_eps_lr_lrs, bounds=bounds3, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds3) * np.log(T) + 2*negLL

        #store in dataframe
        M3eps_lr_lrs.at[i, 'eps'] = param_fits[0]
        M3eps_lr_lrs.at[i, 'lr'] = param_fits[1]
        M3eps_lr_lrs.at[i, 'lrs'] = param_fits[2]
        M3eps_lr_lrs.at[i, 'w'] = param_fits[3]
        M3eps_lr_lrs.at[i, 'negLL'] = negLL
        M3eps_lr_lrs.at[i, 'BIC'] = BIC

        M3BIC_all.at[i, 'eps-lr-lrs'] = BIC
        title_excel = os.path.join(save_dir, f'HaS1var_M3_3modelrecov_eps_lr_lrs.xlsx')
        M3eps_lr_lrs.to_excel(title_excel, index=False)

            


        #model 4 [eps-lr-uc-lrs]
        #eps, lr, uc, lrs, w = params

        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_lr_uc_lrs, bounds=bounds4, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds4) * np.log(T) + 2*negLL

        M3eps_lr_uc_lrs.at[i, 'eps'] = param_fits[0]
        M3eps_lr_uc_lrs.at[i, 'lr'] = param_fits[1]
        M3eps_lr_uc_lrs.at[i, 'uc'] = param_fits[2]
        M3eps_lr_uc_lrs.at[i, 'lrs'] = param_fits[3]
        M3eps_lr_uc_lrs.at[i, 'w'] = param_fits[4]
        M3eps_lr_uc_lrs.at[i, 'negLL'] = negLL
        M3eps_lr_uc_lrs.at[i, 'BIC'] = BIC

        M3BIC_all.at[i, 'eps-lr-uc-lrs'] = BIC
        title_excel = os.path.join(save_dir, f'HaS1var_M3_4modelrecov_eps_lr_uc_lrs.xlsx')
        M3eps_lr_uc_lrs.to_excel(title_excel, index=False)


            

        #model 5 [eps-lrpos-lrneg]
        #eps, lrpos, lrneg = params
        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_2lr, bounds=bounds5, args=(k,r), strategy=strategy)


        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds5) * np.log(T) + 2*negLL


        #store in dataframe
        M3eps_2lr.at[i, 'eps'] = param_fits[0]
        M3eps_2lr.at[i, 'lrpos'] = param_fits[1]
        M3eps_2lr.at[i, 'lrneg'] = param_fits[2]
        M3eps_2lr.at[i, 'negLL'] = negLL
        M3eps_2lr.at[i, 'BIC'] = BIC

        M3BIC_all.at[i, 'eps-2lr'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1var_M3_5modelrecov_eps_2lr.xlsx')
        M3eps_2lr.to_excel(title_excel, index=False)




        #model 6 [eps-lrpos-lrneg-uc]
        #eps, lrpos, lrneg, uc = params
        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_2lr_uc, bounds=bounds6, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds6) * np.log(T) + 2*negLL

        #store in dataframe
        M3eps_2lr_uc.at[i, 'eps'] = param_fits[0]
        M3eps_2lr_uc.at[i, 'lrpos'] = param_fits[1]
        M3eps_2lr_uc.at[i, 'lrneg'] = param_fits[2]   
        M3eps_2lr_uc.at[i, 'uc'] = param_fits[3]
        M3eps_2lr_uc.at[i, 'negLL'] = negLL
        M3eps_2lr_uc.at[i, 'BIC'] = BIC
        M3BIC_all.at[i, 'eps-2lr-uc'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1var_M3_6modelrecov_eps_2lr_uc.xlsx')
        M3eps_2lr_uc.to_excel(title_excel, index=False)





        #model 7 [eps-lrpos-lrneg-lrs]
        #eps, lrpos, lrneg, lrs, w = params

        negLL = np.inf #initialize negative log likelihood

        result = differential_evolution(negll_RW_eps_2lr_lrs, bounds=bounds7, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds7) * np.log(T) + 2*negLL

        #store in dataframe
        M3eps_2lr_lrs.at[i, 'eps'] = param_fits[0]
        M3eps_2lr_lrs.at[i, 'lrpos'] = param_fits[1]
        M3eps_2lr_lrs.at[i, 'lrneg'] = param_fits[2]
        M3eps_2lr_lrs.at[i, 'lrs'] = param_fits[3]
        M3eps_2lr_lrs.at[i, 'w'] = param_fits[4]
        M3eps_2lr_lrs.at[i, 'negLL'] = negLL
        M3eps_2lr_lrs.at[i, 'BIC'] = BIC
        M3BIC_all.at[i, 'eps-2lr-lrs'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1var_M3_7modelrecov_eps_2lr_lrs.xlsx')
        M3eps_2lr_lrs.to_excel(title_excel, index=False)




        #model 8 [eps-lrpos-lrneg-uc-lrs]
        #eps, lrpos, lrneg, uc, lrs, w = params
        negLL = np.inf #initialize negative log likelihood

        #eps, lrpos, lrneg, lrs, w = params
        result = differential_evolution(negll_RW_eps_2lr_uc_lrs, bounds=bounds8, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds8) * np.log(T) + 2*negLL

        #store in dataframe
        M3eps_2lr_uc_lrs.at[i, 'eps'] = param_fits[0]
        M3eps_2lr_uc_lrs.at[i, 'lrpos'] = param_fits[1]
        M3eps_2lr_uc_lrs.at[i, 'lrneg'] = param_fits[2]
        M3eps_2lr_uc_lrs.at[i, 'uc'] = param_fits[3]
        M3eps_2lr_uc_lrs.at[i, 'lrs'] = param_fits[4]
        M3eps_2lr_uc_lrs.at[i, 'w'] = param_fits[5]
        M3eps_2lr_uc_lrs.at[i, 'negLL'] = negLL
        M3eps_2lr_uc_lrs.at[i, 'BIC'] = BIC
        M3BIC_all.at[i, 'eps-2lr-uc-lrs'] = BIC
        title_excel = os.path.join(save_dir, f'HaS1var_M3_8modelrecov_eps_2lr_uc_lrs.xlsx')
        M3eps_2lr_uc_lrs.to_excel(title_excel, index=False)

        title_excel = os.path.join(save_dir, f'HaS1var_M3_modelrecov_BIC_all.xlsx')
        M3BIC_all.to_excel(title_excel, index=False)   

    

    #########################################################################################
    #simulate model 4 [eps-lr-uc-lrs]
    #########################################################################################
    sample_dir = os.path.join(next_dir2, '4_HaS1var_modelfit_eps_lr_uc_lrs.xlsx')
    sample_pool = pd.read_excel(sample_dir)


    M4BIC_all = pd.DataFrame(index=range(0, 1), columns=['eps-lr', 'eps-lr-uc', 'eps-lr-lrs', 'eps-lr-uc-lrs', 'eps-2lr', 'eps-2lr-uc', 'eps-2lr-lrs', 'eps-2lr-uc-lrs'])
    M4sampled_eps_lr_uc_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr','uc', 'lrs', 'w'])
    M4eps_lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'negLL', 'BIC'])
    M4eps_lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'uc', 'negLL', 'BIC'])
    M4eps_lr_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'lrs', 'w', 'negLL', 'BIC'])
    M4eps_lr_uc_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'uc', 'lrs', 'w', 'negLL', 'BIC'])    
    M4eps_2lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'negLL', 'BIC'])    
    M4eps_2lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'uc', 'negLL', 'BIC'])
    M4eps_2lr_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'lrs', 'w', 'negLL', 'BIC'])
    M4eps_2lr_uc_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'uc', 'lrs', 'w', 'negLL', 'BIC'])

    for i in range(recov_amount):
        print('M4:', i+1)
        ri = np.random.randint(0, len(sample_pool['eps']))
        # parameter selection from estimated values from has data
        true_eps = float(sample_pool['eps'].iloc[ri])
        ri = np.random.randint(0, len(sample_pool['eps']))
        true_lr = float(sample_pool['lr'].iloc[ri])
        ri = np.random.randint(0, len(sample_pool['eps']))
        true_uc = float(sample_pool['uc'].iloc[ri])
        ri = np.random.randint(0, len(sample_pool['eps']))
        true_lrs = float(sample_pool['lrs'].iloc[ri])
        ri = np.random.randint(0, len(sample_pool['eps']))
        true_w = float(sample_pool['w'].iloc[ri])
        M4sampled_eps_lr_uc_lrs.at[i, 'eps'] = true_eps
        M4sampled_eps_lr_uc_lrs.at[i, 'lr'] = true_lr
        M4sampled_eps_lr_uc_lrs.at[i, 'uc'] = true_uc
        M4sampled_eps_lr_uc_lrs.at[i, 'lrs'] = true_lrs
        M4sampled_eps_lr_uc_lrs.at[i, 'w'] = true_w
        title_excel = os.path.join(save_dir, f'HaS1var_M4_modelrecov_eps_lr_uc_lrs_sampled.xlsx')
        M4sampled_eps_lr_uc_lrs.to_excel(title_excel, index=False)

        k, r, Q_k_stored = RW_eps_lr_uc_lrs_adversarial(eps=true_eps, lr=true_lr, uc=true_uc, lrs= true_lrs, w=true_w, T=T, Q_int=Q_int)



        #model 1 [eps-lr]
        #eps, lr = params
        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_lr, bounds=bounds1, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds1) * np.log(T) + 2*negLL

        #store in dataframe
        M4eps_lr.at[i, 'eps'] = param_fits[0]
        M4eps_lr.at[i, 'lr'] = param_fits[1]
        M4eps_lr.at[i, 'negLL'] = negLL
        M4eps_lr.at[i, 'BIC'] = BIC

        M4BIC_all.at[i, 'eps-lr'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1var_M4_1modelrecov_eps_lr.xlsx')
        M4eps_lr.to_excel(title_excel, index=False)



        #model 2 [eps-lr-uc]
        #eps, lr, uc = params
        negLL = np.inf #initialize negative log likelihood

        result = differential_evolution(negll_RW_eps_lr_uc, bounds=bounds2, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds2) * np.log(T) + 2*negLL

        #store in dataframe
        M4eps_lr_uc.at[i, 'eps'] = param_fits[0]
        M4eps_lr_uc.at[i, 'lr'] = param_fits[1]
        M4eps_lr_uc.at[i, 'uc'] = param_fits[2]
        M4eps_lr_uc.at[i, 'negLL'] = negLL
        M4eps_lr_uc.at[i, 'BIC'] = BIC
        M4BIC_all.at[i, 'eps-lr-uc'] = BIC    


        title_excel = os.path.join(save_dir, f'HaS1var_M4_2modelrecov_eps_lr_uc.xlsx')
        M4eps_lr_uc.to_excel(title_excel, index=False)



        #model 3 [eps-lr-lrs]
        #eps, lr, lrs, w = params
        negLL = np.inf #initialize negative log likelihood

        result = differential_evolution(negll_RW_eps_lr_lrs, bounds=bounds3, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds3) * np.log(T) + 2*negLL

        #store in dataframe
        M4eps_lr_lrs.at[i, 'eps'] = param_fits[0]
        M4eps_lr_lrs.at[i, 'lr'] = param_fits[1]
        M4eps_lr_lrs.at[i, 'lrs'] = param_fits[2]
        M4eps_lr_lrs.at[i, 'w'] = param_fits[3]
        M4eps_lr_lrs.at[i, 'negLL'] = negLL
        M4eps_lr_lrs.at[i, 'BIC'] = BIC

        M4BIC_all.at[i, 'eps-lr-lrs'] = BIC
        title_excel = os.path.join(save_dir, f'HaS1var_M4_3modelrecov_eps_lr_lrs.xlsx')
        M4eps_lr_lrs.to_excel(title_excel, index=False)

            


        #model 4 [eps-lr-uc-lrs]
        #eps, lr, uc, lrs, w = params

        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_lr_uc_lrs, bounds=bounds4, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds4) * np.log(T) + 2*negLL

        M4eps_lr_uc_lrs.at[i, 'eps'] = param_fits[0]
        M4eps_lr_uc_lrs.at[i, 'lr'] = param_fits[1]
        M4eps_lr_uc_lrs.at[i, 'uc'] = param_fits[2]
        M4eps_lr_uc_lrs.at[i, 'lrs'] = param_fits[3]
        M4eps_lr_uc_lrs.at[i, 'w'] = param_fits[4]
        M4eps_lr_uc_lrs.at[i, 'negLL'] = negLL
        M4eps_lr_uc_lrs.at[i, 'BIC'] = BIC

        M4BIC_all.at[i, 'eps-lr-uc-lrs'] = BIC
        title_excel = os.path.join(save_dir, f'HaS1var_M4_4modelrecov_eps_lr_uc_lrs.xlsx')
        M4eps_lr_uc_lrs.to_excel(title_excel, index=False)


            

        #model 5 [eps-lrpos-lrneg]
        #eps, lrpos, lrneg = params
        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_2lr, bounds=bounds5, args=(k,r), strategy=strategy)


        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds5) * np.log(T) + 2*negLL


        #store in dataframe
        M4eps_2lr.at[i, 'eps'] = param_fits[0]
        M4eps_2lr.at[i, 'lrpos'] = param_fits[1]
        M4eps_2lr.at[i, 'lrneg'] = param_fits[2]
        M4eps_2lr.at[i, 'negLL'] = negLL
        M4eps_2lr.at[i, 'BIC'] = BIC

        M4BIC_all.at[i, 'eps-2lr'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1var_M4_5modelrecov_eps_2lr.xlsx')
        M4eps_2lr.to_excel(title_excel, index=False)




        #model 6 [eps-lrpos-lrneg-uc]
        #eps, lrpos, lrneg, uc = params
        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_2lr_uc, bounds=bounds6, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds6) * np.log(T) + 2*negLL

        #store in dataframe
        M4eps_2lr_uc.at[i, 'eps'] = param_fits[0]
        M4eps_2lr_uc.at[i, 'lrpos'] = param_fits[1]
        M4eps_2lr_uc.at[i, 'lrneg'] = param_fits[2]   
        M4eps_2lr_uc.at[i, 'uc'] = param_fits[3]
        M4eps_2lr_uc.at[i, 'negLL'] = negLL
        M4eps_2lr_uc.at[i, 'BIC'] = BIC
        M4BIC_all.at[i, 'eps-2lr-uc'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1var_M4_6modelrecov_eps_2lr_uc.xlsx')
        M4eps_2lr_uc.to_excel(title_excel, index=False)





        #model 7 [eps-lrpos-lrneg-lrs]
        #eps, lrpos, lrneg, lrs, w = params

        negLL = np.inf #initialize negative log likelihood

        result = differential_evolution(negll_RW_eps_2lr_lrs, bounds=bounds7, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds7) * np.log(T) + 2*negLL

        #store in dataframe
        M4eps_2lr_lrs.at[i, 'eps'] = param_fits[0]
        M4eps_2lr_lrs.at[i, 'lrpos'] = param_fits[1]
        M4eps_2lr_lrs.at[i, 'lrneg'] = param_fits[2]
        M4eps_2lr_lrs.at[i, 'lrs'] = param_fits[3]
        M4eps_2lr_lrs.at[i, 'w'] = param_fits[4]
        M4eps_2lr_lrs.at[i, 'negLL'] = negLL
        M4eps_2lr_lrs.at[i, 'BIC'] = BIC
        M4BIC_all.at[i, 'eps-2lr-lrs'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1var_M4_7modelrecov_eps_2lr_lrs.xlsx')
        M4eps_2lr_lrs.to_excel(title_excel, index=False)




        #model 8 [eps-lrpos-lrneg-uc-lrs]
        #eps, lrpos, lrneg, uc, lrs, w = params
        negLL = np.inf #initialize negative log likelihood

        #eps, lrpos, lrneg, lrs, w = params
        result = differential_evolution(negll_RW_eps_2lr_uc_lrs, bounds=bounds8, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds8) * np.log(T) + 2*negLL

        #store in dataframe
        M4eps_2lr_uc_lrs.at[i, 'eps'] = param_fits[0]
        M4eps_2lr_uc_lrs.at[i, 'lrpos'] = param_fits[1]
        M4eps_2lr_uc_lrs.at[i, 'lrneg'] = param_fits[2]
        M4eps_2lr_uc_lrs.at[i, 'uc'] = param_fits[3]
        M4eps_2lr_uc_lrs.at[i, 'lrs'] = param_fits[4]
        M4eps_2lr_uc_lrs.at[i, 'w'] = param_fits[5]
        M4eps_2lr_uc_lrs.at[i, 'negLL'] = negLL
        M4eps_2lr_uc_lrs.at[i, 'BIC'] = BIC
        M4BIC_all.at[i, 'eps-2lr-uc-lrs'] = BIC
        title_excel = os.path.join(save_dir, f'HaS1var_M4_8modelrecov_eps_2lr_uc_lrs.xlsx')
        M4eps_2lr_uc_lrs.to_excel(title_excel, index=False)

        title_excel = os.path.join(save_dir, f'HaS1var_M4_modelrecov_BIC_all.xlsx')
        M4BIC_all.to_excel(title_excel, index=False)   

        

    
    #########################################################################################
    #simulate model 5 [eps-2lr]
    #########################################################################################
    sample_dir = os.path.join(next_dir2, '5_HaS1var_modelfit_eps_2lr.xlsx')
    sample_pool = pd.read_excel(sample_dir)


    M5BIC_all = pd.DataFrame(index=range(0, 1), columns=['eps-lr', 'eps-lr-uc', 'eps-lr-lrs', 'eps-lr-uc-lrs', 'eps-2lr', 'eps-2lr-uc', 'eps-2lr-lrs', 'eps-2lr-uc-lrs'])
    M5sampled_eps_2lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos','lrneg'])
    M5eps_lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'negLL', 'BIC'])
    M5eps_lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'uc', 'negLL', 'BIC'])
    M5eps_lr_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'lrs', 'w', 'negLL', 'BIC'])
    M5eps_lr_uc_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'uc', 'lrs', 'w', 'negLL', 'BIC'])    
    M5eps_2lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'negLL', 'BIC'])    
    M5eps_2lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'uc', 'negLL', 'BIC'])
    M5eps_2lr_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'lrs', 'w', 'negLL', 'BIC'])
    M5eps_2lr_uc_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'uc', 'lrs', 'w', 'negLL', 'BIC'])


    for i in range(recov_amount):
        print('M5:', i+1)
        ri = np.random.randint(0, len(sample_pool['eps']))
        # parameter selection from estimated values from has data
        true_eps = float(sample_pool['eps'].iloc[ri])
        ri = np.random.randint(0, len(sample_pool['eps']))
        true_lrpos = float(sample_pool['lrpos'].iloc[ri])
        true_lrneg = float(sample_pool['lrneg'].iloc[ri])
        M5sampled_eps_2lr.at[i, 'eps'] = true_eps
        M5sampled_eps_2lr.at[i, 'lrpos'] = true_lrpos
        M5sampled_eps_2lr.at[i, 'lrneg'] = true_lrneg
        print(true_lrpos, true_lrneg)

        title_excel = os.path.join(save_dir, f'HaS1var_M5_modelrecov_eps_2lr_sampled.xlsx')
        M5sampled_eps_2lr.to_excel(title_excel, index=False)    

        k, r, Q_k_stored = RW_eps_2lr_adversarial(eps=true_eps, lrpos=true_lrpos, lrneg=true_lrneg, T=T, Q_int=Q_int)



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

        title_excel = os.path.join(save_dir, f'HaS1var_M5_1modelrecov_eps_lr.xlsx')
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


        title_excel = os.path.join(save_dir, f'HaS1var_M5_2modelrecov_eps_lr_uc.xlsx')
        M5eps_lr_uc.to_excel(title_excel, index=False)



        #model 3 [eps-lr-lrs]
        #eps, lr, lrs, w = params
        negLL = np.inf #initialize negative log likelihood

        result = differential_evolution(negll_RW_eps_lr_lrs, bounds=bounds3, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds3) * np.log(T) + 2*negLL

        #store in dataframe
        M5eps_lr_lrs.at[i, 'eps'] = param_fits[0]
        M5eps_lr_lrs.at[i, 'lr'] = param_fits[1]
        M5eps_lr_lrs.at[i, 'lrs'] = param_fits[2]
        M5eps_lr_lrs.at[i, 'w'] = param_fits[3]
        M5eps_lr_lrs.at[i, 'negLL'] = negLL
        M5eps_lr_lrs.at[i, 'BIC'] = BIC

        M5BIC_all.at[i, 'eps-lr-lrs'] = BIC
        title_excel = os.path.join(save_dir, f'HaS1var_M5_3modelrecov_eps_lr_lrs.xlsx')
        M5eps_lr_lrs.to_excel(title_excel, index=False)

            


        #model 4 [eps-lr-uc-lrs]
        #eps, lr, uc, lrs, w = params

        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_lr_uc_lrs, bounds=bounds4, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds4) * np.log(T) + 2*negLL

        M5eps_lr_uc_lrs.at[i, 'eps'] = param_fits[0]
        M5eps_lr_uc_lrs.at[i, 'lr'] = param_fits[1]
        M5eps_lr_uc_lrs.at[i, 'uc'] = param_fits[2]
        M5eps_lr_uc_lrs.at[i, 'lrs'] = param_fits[3]
        M5eps_lr_uc_lrs.at[i, 'w'] = param_fits[4]
        M5eps_lr_uc_lrs.at[i, 'negLL'] = negLL
        M5eps_lr_uc_lrs.at[i, 'BIC'] = BIC

        M5BIC_all.at[i, 'eps-lr-uc-lrs'] = BIC
        title_excel = os.path.join(save_dir, f'HaS1var_M5_4modelrecov_eps_lr_uc_lrs.xlsx')
        M5eps_lr_uc_lrs.to_excel(title_excel, index=False)


            

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

        title_excel = os.path.join(save_dir, f'HaS1var_M5_5modelrecov_eps_2lr.xlsx')
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

        title_excel = os.path.join(save_dir, f'HaS1var_M5_6modelrecov_eps_2lr_uc.xlsx')
        M5eps_2lr_uc.to_excel(title_excel, index=False)





        #model 7 [eps-lrpos-lrneg-lrs]
        #eps, lrpos, lrneg, lrs, w = params

        negLL = np.inf #initialize negative log likelihood

        result = differential_evolution(negll_RW_eps_2lr_lrs, bounds=bounds7, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds7) * np.log(T) + 2*negLL

        #store in dataframe
        M5eps_2lr_lrs.at[i, 'eps'] = param_fits[0]
        M5eps_2lr_lrs.at[i, 'lrpos'] = param_fits[1]
        M5eps_2lr_lrs.at[i, 'lrneg'] = param_fits[2]
        M5eps_2lr_lrs.at[i, 'lrs'] = param_fits[3]
        M5eps_2lr_lrs.at[i, 'w'] = param_fits[4]
        M5eps_2lr_lrs.at[i, 'negLL'] = negLL
        M5eps_2lr_lrs.at[i, 'BIC'] = BIC
        M5BIC_all.at[i, 'eps-2lr-lrs'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1var_M5_7modelrecov_eps_2lr_lrs.xlsx')
        M5eps_2lr_lrs.to_excel(title_excel, index=False)




        #model 8 [eps-lrpos-lrneg-uc-lrs]
        #eps, lrpos, lrneg, uc, lrs, w = params
        negLL = np.inf #initialize negative log likelihood

        #eps, lrpos, lrneg, lrs, w = params
        result = differential_evolution(negll_RW_eps_2lr_uc_lrs, bounds=bounds8, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds8) * np.log(T) + 2*negLL

        #store in dataframe
        M5eps_2lr_uc_lrs.at[i, 'eps'] = param_fits[0]
        M5eps_2lr_uc_lrs.at[i, 'lrpos'] = param_fits[1]
        M5eps_2lr_uc_lrs.at[i, 'lrneg'] = param_fits[2]
        M5eps_2lr_uc_lrs.at[i, 'uc'] = param_fits[3]
        M5eps_2lr_uc_lrs.at[i, 'lrs'] = param_fits[4]
        M5eps_2lr_uc_lrs.at[i, 'w'] = param_fits[5]
        M5eps_2lr_uc_lrs.at[i, 'negLL'] = negLL
        M5eps_2lr_uc_lrs.at[i, 'BIC'] = BIC
        M5BIC_all.at[i, 'eps-2lr-uc-lrs'] = BIC
        title_excel = os.path.join(save_dir, f'HaS1var_M5_8modelrecov_eps_2lr_uc_lrs.xlsx')
        M5eps_2lr_uc_lrs.to_excel(title_excel, index=False)

        title_excel = os.path.join(save_dir, f'HaS1var_M5_modelrecov_BIC_all.xlsx')
        M5BIC_all.to_excel(title_excel, index=False)   

    '''
    #########################################################################################
    #simulate model 6 [eps-2lr-uc]
    #########################################################################################
    sample_dir = os.path.join(next_dir2, '6_HaS1var_modelfit_eps_2lr_uc.xlsx')
    sample_pool = pd.read_excel(sample_dir)


    M6BIC_all = pd.DataFrame(index=range(0, 1), columns=['eps-lr', 'eps-lr-uc', 'eps-lr-lrs', 'eps-lr-uc-lrs', 'eps-2lr', 'eps-2lr-uc', 'eps-2lr-lrs', 'eps-2lr-uc-lrs'])
    M6sampled_eps_2lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos','lrneg', 'uc'])
    M6eps_lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'negLL', 'BIC'])
    M6eps_lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'uc', 'negLL', 'BIC'])
    M6eps_lr_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'lrs', 'w', 'negLL', 'BIC'])
    M6eps_lr_uc_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'uc', 'lrs', 'w', 'negLL', 'BIC'])    
    M6eps_2lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'negLL', 'BIC'])    
    M6eps_2lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'uc', 'negLL', 'BIC'])
    M6eps_2lr_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'lrs', 'w', 'negLL', 'BIC'])
    M6eps_2lr_uc_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'uc', 'lrs', 'w', 'negLL', 'BIC'])

    for i in range(recov_amount):
        print('M6:', i+1)
        ri = np.random.randint(0, len(sample_pool['eps']))
        # parameter selection from estimated values from has data
        true_eps = float(sample_pool['eps'].iloc[ri])
        ri = np.random.randint(0, len(sample_pool['eps']))
        true_lrpos = float(sample_pool['lrpos'].iloc[ri])
        true_lrneg = float(sample_pool['lrneg'].iloc[ri])
        ri = np.random.randint(0, len(sample_pool['eps']))
        true_uc = float(sample_pool['uc'].iloc[ri])
        M6sampled_eps_2lr_uc.at[i, 'eps'] = true_eps
        M6sampled_eps_2lr_uc.at[i, 'lrpos'] = true_lrpos
        M6sampled_eps_2lr_uc.at[i, 'lrneg'] = true_lrneg
        M6sampled_eps_2lr_uc.at[i, 'uc'] = true_uc
        print(true_lrpos, true_lrneg)

        title_excel = os.path.join(save_dir, f'HaS1var_M6_modelrecov_eps_2lr_uc_sampled.xlsx')
        M6sampled_eps_2lr_uc.to_excel(title_excel, index=False)

        k, r, Q_k_stored = RW_eps_2lr_uc_adversarial(eps=true_eps, lrpos=true_lrpos, lrneg=true_lrneg, uc=true_uc, T=T, Q_int=Q_int)



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

        title_excel = os.path.join(save_dir, f'HaS1var_M6_1modelrecov_eps_lr.xlsx')
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


        title_excel = os.path.join(save_dir, f'HaS1var_M6_2modelrecov_eps_lr_uc.xlsx')
        M6eps_lr_uc.to_excel(title_excel, index=False)



        #model 3 [eps-lr-lrs]
        #eps, lr, lrs, w = params
        negLL = np.inf #initialize negative log likelihood

        result = differential_evolution(negll_RW_eps_lr_lrs, bounds=bounds3, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds3) * np.log(T) + 2*negLL

        #store in dataframe
        M6eps_lr_lrs.at[i, 'eps'] = param_fits[0]
        M6eps_lr_lrs.at[i, 'lr'] = param_fits[1]
        M6eps_lr_lrs.at[i, 'lrs'] = param_fits[2]
        M6eps_lr_lrs.at[i, 'w'] = param_fits[3]
        M6eps_lr_lrs.at[i, 'negLL'] = negLL
        M6eps_lr_lrs.at[i, 'BIC'] = BIC

        M6BIC_all.at[i, 'eps-lr-lrs'] = BIC
        title_excel = os.path.join(save_dir, f'HaS1var_M6_3modelrecov_eps_lr_lrs.xlsx')
        M6eps_lr_lrs.to_excel(title_excel, index=False)

            


        #model 4 [eps-lr-uc-lrs]
        #eps, lr, uc, lrs, w = params

        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_lr_uc_lrs, bounds=bounds4, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds4) * np.log(T) + 2*negLL

        M6eps_lr_uc_lrs.at[i, 'eps'] = param_fits[0]
        M6eps_lr_uc_lrs.at[i, 'lr'] = param_fits[1]
        M6eps_lr_uc_lrs.at[i, 'uc'] = param_fits[2]
        M6eps_lr_uc_lrs.at[i, 'lrs'] = param_fits[3]
        M6eps_lr_uc_lrs.at[i, 'w'] = param_fits[4]
        M6eps_lr_uc_lrs.at[i, 'negLL'] = negLL
        M6eps_lr_uc_lrs.at[i, 'BIC'] = BIC

        M6BIC_all.at[i, 'eps-lr-uc-lrs'] = BIC
        title_excel = os.path.join(save_dir, f'HaS1var_M6_4modelrecov_eps_lr_uc_lrs.xlsx')
        M6eps_lr_uc_lrs.to_excel(title_excel, index=False)


            

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

        title_excel = os.path.join(save_dir, f'HaS1var_M6_5modelrecov_eps_2lr.xlsx')
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

        title_excel = os.path.join(save_dir, f'HaS1var_M6_6modelrecov_eps_2lr_uc.xlsx')
        M6eps_2lr_uc.to_excel(title_excel, index=False)





        #model 7 [eps-lrpos-lrneg-lrs]
        #eps, lrpos, lrneg, lrs, w = params

        negLL = np.inf #initialize negative log likelihood

        result = differential_evolution(negll_RW_eps_2lr_lrs, bounds=bounds7, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds7) * np.log(T) + 2*negLL

        #store in dataframe
        M6eps_2lr_lrs.at[i, 'eps'] = param_fits[0]
        M6eps_2lr_lrs.at[i, 'lrpos'] = param_fits[1]
        M6eps_2lr_lrs.at[i, 'lrneg'] = param_fits[2]
        M6eps_2lr_lrs.at[i, 'lrs'] = param_fits[3]
        M6eps_2lr_lrs.at[i, 'w'] = param_fits[4]
        M6eps_2lr_lrs.at[i, 'negLL'] = negLL
        M6eps_2lr_lrs.at[i, 'BIC'] = BIC
        M6BIC_all.at[i, 'eps-2lr-lrs'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1var_M6_7modelrecov_eps_2lr_lrs.xlsx')
        M6eps_2lr_lrs.to_excel(title_excel, index=False)




        #model 8 [eps-lrpos-lrneg-uc-lrs]
        #eps, lrpos, lrneg, uc, lrs, w = params
        negLL = np.inf #initialize negative log likelihood

        #eps, lrpos, lrneg, lrs, w = params
        result = differential_evolution(negll_RW_eps_2lr_uc_lrs, bounds=bounds8, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds8) * np.log(T) + 2*negLL

        #store in dataframe
        M6eps_2lr_uc_lrs.at[i, 'eps'] = param_fits[0]
        M6eps_2lr_uc_lrs.at[i, 'lrpos'] = param_fits[1]
        M6eps_2lr_uc_lrs.at[i, 'lrneg'] = param_fits[2]
        M6eps_2lr_uc_lrs.at[i, 'uc'] = param_fits[3]
        M6eps_2lr_uc_lrs.at[i, 'lrs'] = param_fits[4]
        M6eps_2lr_uc_lrs.at[i, 'w'] = param_fits[5]
        M6eps_2lr_uc_lrs.at[i, 'negLL'] = negLL
        M6eps_2lr_uc_lrs.at[i, 'BIC'] = BIC
        M6BIC_all.at[i, 'eps-2lr-uc-lrs'] = BIC
        title_excel = os.path.join(save_dir, f'HaS1var_M6_8modelrecov_eps_2lr_uc_lrs.xlsx')
        M6eps_2lr_uc_lrs.to_excel(title_excel, index=False)

        title_excel = os.path.join(save_dir, f'HaS1var_M6_modelrecov_BIC_all.xlsx')
        M6BIC_all.to_excel(title_excel, index=False)   

    
    '''
    #########################################################################################
    #simulate model 7 [eps-2lr-lrs]
    #########################################################################################
    sample_dir = os.path.join(next_dir2, '7_HaS1var_modelfit_eps_2lr_lrs.xlsx')
    sample_pool = pd.read_excel(sample_dir)


    M7BIC_all = pd.DataFrame(index=range(0, 1), columns=['eps-lr', 'eps-lr-uc', 'eps-lr-lrs', 'eps-lr-uc-lrs', 'eps-2lr', 'eps-2lr-uc', 'eps-2lr-lrs', 'eps-2lr-uc-lrs'])
    M7sampled_eps_2lr_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos','lrneg', 'lrs', 'w'])
    M7eps_lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'negLL', 'BIC'])
    M7eps_lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'uc', 'negLL', 'BIC'])
    M7eps_lr_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'lrs', 'w', 'negLL', 'BIC'])
    M7eps_lr_uc_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'uc', 'lrs', 'w', 'negLL', 'BIC'])    
    M7eps_2lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'negLL', 'BIC'])    
    M7eps_2lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'uc', 'negLL', 'BIC'])
    M7eps_2lr_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'lrs', 'w', 'negLL', 'BIC'])
    M7eps_2lr_uc_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'uc', 'lrs', 'w', 'negLL', 'BIC'])

    for i in range(recov_amount):
        print('M7:', i+1)
        ri = np.random.randint(0, len(sample_pool['eps']))
        # parameter selection from estimated values from has data
        true_eps = float(sample_pool['eps'].iloc[ri])
        ri = np.random.randint(0, len(sample_pool['eps']))
        true_lrpos = float(sample_pool['lrpos'].iloc[ri])
        true_lrneg = float(sample_pool['lrneg'].iloc[ri])
        ri = np.random.randint(0, len(sample_pool['eps']))
        true_lrs = float(sample_pool['lrs'].iloc[ri])
        ri = np.random.randint(0, len(sample_pool['eps']))
        true_w = float(sample_pool['w'].iloc[ri])
        M7sampled_eps_2lr_lrs.at[i, 'eps'] = true_eps
        M7sampled_eps_2lr_lrs.at[i, 'lrpos'] = true_lrpos
        M7sampled_eps_2lr_lrs.at[i, 'lrneg'] = true_lrneg
        M7sampled_eps_2lr_lrs.at[i, 'lrs'] = true_lrs
        M7sampled_eps_2lr_lrs.at[i, 'w'] = true_w
        print(true_lrpos, true_lrneg)

        title_excel = os.path.join(save_dir, f'HaS1var_M7_modelrecov_eps_2lr_lrs_sampled.xlsx')
        M7sampled_eps_2lr_lrs.to_excel(title_excel, index=False)

        k, r, Q_k_stored = RW_eps_2lr_lrs_adversarial(eps=true_eps, lrpos=true_lrpos, lrneg=true_lrneg, lrs=true_lrs, w=true_w, T=T, Q_int=Q_int)



        #model 1 [eps-lr]
        #eps, lr = params
        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_lr, bounds=bounds1, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds1) * np.log(T) + 2*negLL

        #store in dataframe
        M7eps_lr.at[i, 'eps'] = param_fits[0]
        M7eps_lr.at[i, 'lr'] = param_fits[1]
        M7eps_lr.at[i, 'negLL'] = negLL
        M7eps_lr.at[i, 'BIC'] = BIC

        M7BIC_all.at[i, 'eps-lr'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1var_M7_1modelrecov_eps_lr.xlsx')
        M7eps_lr.to_excel(title_excel, index=False)



        #model 2 [eps-lr-uc]
        #eps, lr, uc = params
        negLL = np.inf #initialize negative log likelihood

        result = differential_evolution(negll_RW_eps_lr_uc, bounds=bounds2, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds2) * np.log(T) + 2*negLL

        #store in dataframe
        M7eps_lr_uc.at[i, 'eps'] = param_fits[0]
        M7eps_lr_uc.at[i, 'lr'] = param_fits[1]
        M7eps_lr_uc.at[i, 'uc'] = param_fits[2]
        M7eps_lr_uc.at[i, 'negLL'] = negLL
        M7eps_lr_uc.at[i, 'BIC'] = BIC
        M7BIC_all.at[i, 'eps-lr-uc'] = BIC    


        title_excel = os.path.join(save_dir, f'HaS1var_M7_2modelrecov_eps_lr_uc.xlsx')
        M7eps_lr_uc.to_excel(title_excel, index=False)



        #model 3 [eps-lr-lrs]
        #eps, lr, lrs, w = params
        negLL = np.inf #initialize negative log likelihood

        result = differential_evolution(negll_RW_eps_lr_lrs, bounds=bounds3, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds3) * np.log(T) + 2*negLL

        #store in dataframe
        M7eps_lr_lrs.at[i, 'eps'] = param_fits[0]
        M7eps_lr_lrs.at[i, 'lr'] = param_fits[1]
        M7eps_lr_lrs.at[i, 'lrs'] = param_fits[2]
        M7eps_lr_lrs.at[i, 'w'] = param_fits[3]
        M7eps_lr_lrs.at[i, 'negLL'] = negLL
        M7eps_lr_lrs.at[i, 'BIC'] = BIC

        M7BIC_all.at[i, 'eps-lr-lrs'] = BIC
        title_excel = os.path.join(save_dir, f'HaS1var_M7_3modelrecov_eps_lr_lrs.xlsx')
        M7eps_lr_lrs.to_excel(title_excel, index=False)

            


        #model 4 [eps-lr-uc-lrs]
        #eps, lr, uc, lrs, w = params

        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_lr_uc_lrs, bounds=bounds4, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds4) * np.log(T) + 2*negLL

        M7eps_lr_uc_lrs.at[i, 'eps'] = param_fits[0]
        M7eps_lr_uc_lrs.at[i, 'lr'] = param_fits[1]
        M7eps_lr_uc_lrs.at[i, 'uc'] = param_fits[2]
        M7eps_lr_uc_lrs.at[i, 'lrs'] = param_fits[3]
        M7eps_lr_uc_lrs.at[i, 'w'] = param_fits[4]
        M7eps_lr_uc_lrs.at[i, 'negLL'] = negLL
        M7eps_lr_uc_lrs.at[i, 'BIC'] = BIC

        M7BIC_all.at[i, 'eps-lr-uc-lrs'] = BIC
        title_excel = os.path.join(save_dir, f'HaS1var_M7_4modelrecov_eps_lr_uc_lrs.xlsx')
        M7eps_lr_uc_lrs.to_excel(title_excel, index=False)


            

        #model 5 [eps-lrpos-lrneg]
        #eps, lrpos, lrneg = params
        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_2lr, bounds=bounds5, args=(k,r), strategy=strategy)


        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds5) * np.log(T) + 2*negLL


        #store in dataframe
        M7eps_2lr.at[i, 'eps'] = param_fits[0]
        M7eps_2lr.at[i, 'lrpos'] = param_fits[1]
        M7eps_2lr.at[i, 'lrneg'] = param_fits[2]
        M7eps_2lr.at[i, 'negLL'] = negLL
        M7eps_2lr.at[i, 'BIC'] = BIC

        M7BIC_all.at[i, 'eps-2lr'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1var_M7_5modelrecov_eps_2lr.xlsx')
        M7eps_2lr.to_excel(title_excel, index=False)




        #model 6 [eps-lrpos-lrneg-uc]
        #eps, lrpos, lrneg, uc = params
        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_2lr_uc, bounds=bounds6, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds6) * np.log(T) + 2*negLL

        #store in dataframe
        M7eps_2lr_uc.at[i, 'eps'] = param_fits[0]
        M7eps_2lr_uc.at[i, 'lrpos'] = param_fits[1]
        M7eps_2lr_uc.at[i, 'lrneg'] = param_fits[2]   
        M7eps_2lr_uc.at[i, 'uc'] = param_fits[3]
        M7eps_2lr_uc.at[i, 'negLL'] = negLL
        M7eps_2lr_uc.at[i, 'BIC'] = BIC
        M7BIC_all.at[i, 'eps-2lr-uc'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1var_M7_6modelrecov_eps_2lr_uc.xlsx')
        M7eps_2lr_uc.to_excel(title_excel, index=False)





        #model 7 [eps-lrpos-lrneg-lrs]
        #eps, lrpos, lrneg, lrs, w = params

        negLL = np.inf #initialize negative log likelihood

        result = differential_evolution(negll_RW_eps_2lr_lrs, bounds=bounds7, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds7) * np.log(T) + 2*negLL

        #store in dataframe
        M7eps_2lr_lrs.at[i, 'eps'] = param_fits[0]
        M7eps_2lr_lrs.at[i, 'lrpos'] = param_fits[1]
        M7eps_2lr_lrs.at[i, 'lrneg'] = param_fits[2]
        M7eps_2lr_lrs.at[i, 'lrs'] = param_fits[3]
        M7eps_2lr_lrs.at[i, 'w'] = param_fits[4]
        M7eps_2lr_lrs.at[i, 'negLL'] = negLL
        M7eps_2lr_lrs.at[i, 'BIC'] = BIC
        M7BIC_all.at[i, 'eps-2lr-lrs'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1var_M7_7modelrecov_eps_2lr_lrs.xlsx')
        M7eps_2lr_lrs.to_excel(title_excel, index=False)




        #model 8 [eps-lrpos-lrneg-uc-lrs]
        #eps, lrpos, lrneg, uc, lrs, w = params
        negLL = np.inf #initialize negative log likelihood

        #eps, lrpos, lrneg, lrs, w = params
        result = differential_evolution(negll_RW_eps_2lr_uc_lrs, bounds=bounds8, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds8) * np.log(T) + 2*negLL

        #store in dataframe
        M7eps_2lr_uc_lrs.at[i, 'eps'] = param_fits[0]
        M7eps_2lr_uc_lrs.at[i, 'lrpos'] = param_fits[1]
        M7eps_2lr_uc_lrs.at[i, 'lrneg'] = param_fits[2]
        M7eps_2lr_uc_lrs.at[i, 'uc'] = param_fits[3]
        M7eps_2lr_uc_lrs.at[i, 'lrs'] = param_fits[4]
        M7eps_2lr_uc_lrs.at[i, 'w'] = param_fits[5]
        M7eps_2lr_uc_lrs.at[i, 'negLL'] = negLL
        M7eps_2lr_uc_lrs.at[i, 'BIC'] = BIC
        M7BIC_all.at[i, 'eps-2lr-uc-lrs'] = BIC
        title_excel = os.path.join(save_dir, f'HaS1var_M7_8modelrecov_eps_2lr_uc_lrs.xlsx')
        M7eps_2lr_uc_lrs.to_excel(title_excel, index=False)

        title_excel = os.path.join(save_dir, f'HaS1var_M7_modelrecov_BIC_all.xlsx')
        M7BIC_all.to_excel(title_excel, index=False)   


    

    #########################################################################################
    #simulate model 8 [eps-2lr-uc-lrs]
    #########################################################################################
    sample_dir = os.path.join(next_dir2, '8_HaS1var_modelfit_eps_2lr_uc_lrs.xlsx')
    sample_pool = pd.read_excel(sample_dir)


    M8BIC_all = pd.DataFrame(index=range(0, 1), columns=['eps-lr', 'eps-lr-uc', 'eps-lr-lrs', 'eps-lr-uc-lrs', 'eps-2lr', 'eps-2lr-uc', 'eps-2lr-lrs', 'eps-2lr-uc-lrs'])
    M8sampled_eps_2lr_uc_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos','lrneg','uc', 'lrs', 'w'])
    M8eps_lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'negLL', 'BIC'])
    M8eps_lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'uc', 'negLL', 'BIC'])
    M8eps_lr_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'lrs', 'w', 'negLL', 'BIC'])
    M8eps_lr_uc_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lr', 'uc', 'lrs', 'w', 'negLL', 'BIC'])    
    M8eps_2lr = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'negLL', 'BIC'])    
    M8eps_2lr_uc = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'uc', 'negLL', 'BIC'])
    M8eps_2lr_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'lrs', 'w', 'negLL', 'BIC'])
    M8eps_2lr_uc_lrs = pd.DataFrame(index=range(0, 1), columns=['eps', 'lrpos', 'lrneg', 'uc', 'lrs', 'w', 'negLL', 'BIC'])

    for i in range(recov_amount):
        ri = np.random.randint(0, len(sample_pool['eps']))
        # parameter selection from estimated values from has data
        true_eps = float(sample_pool['eps'].iloc[ri])
        ri = np.random.randint(0, len(sample_pool['eps']))
        true_lrpos = float(sample_pool['lrpos'].iloc[ri])
        true_lrneg = float(sample_pool['lrneg'].iloc[ri])
        ri = np.random.randint(0, len(sample_pool['eps']))
        true_uc = float(sample_pool['uc'].iloc[ri])
        ri = np.random.randint(0, len(sample_pool['eps']))
        true_lrs = float(sample_pool['lrs'].iloc[ri])
        ri = np.random.randint(0, len(sample_pool['eps']))
        true_w = float(sample_pool['w'].iloc[ri])
        M8sampled_eps_2lr_uc_lrs.at[i, 'eps'] = true_eps
        M8sampled_eps_2lr_uc_lrs.at[i, 'lrpos'] = true_lrpos
        M8sampled_eps_2lr_uc_lrs.at[i, 'lrneg'] = true_lrneg
        M8sampled_eps_2lr_uc_lrs.at[i, 'uc'] = true_uc
        M8sampled_eps_2lr_uc_lrs.at[i, 'lrs'] = true_lrs
        M8sampled_eps_2lr_uc_lrs.at[i, 'w'] = true_w
        print(true_lrpos, true_lrneg)

        title_excel = os.path.join(save_dir, f'HaS1var_M8_modelrecov_eps_2lr_uc_lrs_sampled.xlsx')
        M8sampled_eps_2lr_uc_lrs.to_excel(title_excel, index=False)

        k, r, Q_k_stored = RW_eps_2lr_uc_lrs_adversarial(eps=true_eps, lrpos=true_lrpos, lrneg=true_lrneg, uc=true_uc, lrs=true_lrs, w=true_w, T=T, Q_int=Q_int)



        #model 1 [eps-lr]
        #eps, lr = params
        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_lr, bounds=bounds1, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds1) * np.log(T) + 2*negLL

        #store in dataframe
        M8eps_lr.at[i, 'eps'] = param_fits[0]
        M8eps_lr.at[i, 'lr'] = param_fits[1]
        M8eps_lr.at[i, 'negLL'] = negLL
        M8eps_lr.at[i, 'BIC'] = BIC

        M8BIC_all.at[i, 'eps-lr'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1var_M8_1modelrecov_eps_lr.xlsx')
        M8eps_lr.to_excel(title_excel, index=False)



        #model 2 [eps-lr-uc]
        #eps, lr, uc = params
        negLL = np.inf #initialize negative log likelihood

        result = differential_evolution(negll_RW_eps_lr_uc, bounds=bounds2, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds2) * np.log(T) + 2*negLL

        #store in dataframe
        M8eps_lr_uc.at[i, 'eps'] = param_fits[0]
        M8eps_lr_uc.at[i, 'lr'] = param_fits[1]
        M8eps_lr_uc.at[i, 'uc'] = param_fits[2]
        M8eps_lr_uc.at[i, 'negLL'] = negLL
        M8eps_lr_uc.at[i, 'BIC'] = BIC
        M8BIC_all.at[i, 'eps-lr-uc'] = BIC    


        title_excel = os.path.join(save_dir, f'HaS1var_M8_2modelrecov_eps_lr_uc.xlsx')
        M8eps_lr_uc.to_excel(title_excel, index=False)



        #model 3 [eps-lr-lrs]
        #eps, lr, lrs, w = params
        negLL = np.inf #initialize negative log likelihood

        result = differential_evolution(negll_RW_eps_lr_lrs, bounds=bounds3, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds3) * np.log(T) + 2*negLL

        #store in dataframe
        M8eps_lr_lrs.at[i, 'eps'] = param_fits[0]
        M8eps_lr_lrs.at[i, 'lr'] = param_fits[1]
        M8eps_lr_lrs.at[i, 'lrs'] = param_fits[2]
        M8eps_lr_lrs.at[i, 'w'] = param_fits[3]
        M8eps_lr_lrs.at[i, 'negLL'] = negLL
        M8eps_lr_lrs.at[i, 'BIC'] = BIC

        M8BIC_all.at[i, 'eps-lr-lrs'] = BIC
        title_excel = os.path.join(save_dir, f'HaS1var_M8_3modelrecov_eps_lr_lrs.xlsx')
        M8eps_lr_lrs.to_excel(title_excel, index=False)

            


        #model 4 [eps-lr-uc-lrs]
        #eps, lr, uc, lrs, w = params

        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_lr_uc_lrs, bounds=bounds4, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds4) * np.log(T) + 2*negLL

        M8eps_lr_uc_lrs.at[i, 'eps'] = param_fits[0]
        M8eps_lr_uc_lrs.at[i, 'lr'] = param_fits[1]
        M8eps_lr_uc_lrs.at[i, 'uc'] = param_fits[2]
        M8eps_lr_uc_lrs.at[i, 'lrs'] = param_fits[3]
        M8eps_lr_uc_lrs.at[i, 'w'] = param_fits[4]
        M8eps_lr_uc_lrs.at[i, 'negLL'] = negLL
        M8eps_lr_uc_lrs.at[i, 'BIC'] = BIC

        M8BIC_all.at[i, 'eps-lr-uc-lrs'] = BIC
        title_excel = os.path.join(save_dir, f'HaS1var_M8_4modelrecov_eps_lr_uc_lrs.xlsx')
        M8eps_lr_uc_lrs.to_excel(title_excel, index=False)


            

        #model 5 [eps-lrpos-lrneg]
        #eps, lrpos, lrneg = params
        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_2lr, bounds=bounds5, args=(k,r), strategy=strategy)


        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds5) * np.log(T) + 2*negLL


        #store in dataframe
        M8eps_2lr.at[i, 'eps'] = param_fits[0]
        M8eps_2lr.at[i, 'lrpos'] = param_fits[1]
        M8eps_2lr.at[i, 'lrneg'] = param_fits[2]
        M8eps_2lr.at[i, 'negLL'] = negLL
        M8eps_2lr.at[i, 'BIC'] = BIC
        M8BIC_all.at[i, 'eps-2lr'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1var_M8_5modelrecov_eps_2lr.xlsx')
        M8eps_2lr.to_excel(title_excel, index=False)



        #model 6 [eps-lrpos-lrneg-uc]
        #eps, lrpos, lrneg, uc = params
        negLL = np.inf #initialize negative log likelihood
        result = differential_evolution(negll_RW_eps_2lr_uc, bounds=bounds6, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds6) * np.log(T) + 2*negLL

        #store in dataframe
        M8eps_2lr_uc.at[i, 'eps'] = param_fits[0]
        M8eps_2lr_uc.at[i, 'lrpos'] = param_fits[1]
        M8eps_2lr_uc.at[i, 'lrneg'] = param_fits[2]   
        M8eps_2lr_uc.at[i, 'uc'] = param_fits[3]
        M8eps_2lr_uc.at[i, 'negLL'] = negLL
        M8eps_2lr_uc.at[i, 'BIC'] = BIC
        M8BIC_all.at[i, 'eps-2lr-uc'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1var_M8_6modelrecov_eps_2lr_uc.xlsx')
        M8eps_2lr_uc.to_excel(title_excel, index=False)





        #model 7 [eps-lrpos-lrneg-lrs]
        #eps, lrpos, lrneg, lrs, w = params

        negLL = np.inf #initialize negative log likelihood

        result = differential_evolution(negll_RW_eps_2lr_lrs, bounds=bounds7, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds7) * np.log(T) + 2*negLL

        #store in dataframe
        M8eps_2lr_lrs.at[i, 'eps'] = param_fits[0]
        M8eps_2lr_lrs.at[i, 'lrpos'] = param_fits[1]
        M8eps_2lr_lrs.at[i, 'lrneg'] = param_fits[2]
        M8eps_2lr_lrs.at[i, 'lrs'] = param_fits[3]
        M8eps_2lr_lrs.at[i, 'w'] = param_fits[4]
        M8eps_2lr_lrs.at[i, 'negLL'] = negLL
        M8eps_2lr_lrs.at[i, 'BIC'] = BIC
        M8BIC_all.at[i, 'eps-2lr-lrs'] = BIC

        title_excel = os.path.join(save_dir, f'HaS1var_M8_7modelrecov_eps_2lr_lrs.xlsx')
        M8eps_2lr_lrs.to_excel(title_excel, index=False)




        #model 8 [eps-lrpos-lrneg-uc-lrs]
        #eps, lrpos, lrneg, uc, lrs, w = params
        negLL = np.inf #initialize negative log likelihood

        #eps, lrpos, lrneg, lrs, w = params
        result = differential_evolution(negll_RW_eps_2lr_uc_lrs, bounds=bounds8, args=(k,r), strategy=strategy)

        #increasing maxiter and popsize doesn't make LR estimations better
        negLL = result.fun
        param_fits = result.x
        BIC = len(bounds8) * np.log(T) + 2*negLL

        #store in dataframe
        M8eps_2lr_uc_lrs.at[i, 'eps'] = param_fits[0]
        M8eps_2lr_uc_lrs.at[i, 'lrpos'] = param_fits[1]
        M8eps_2lr_uc_lrs.at[i, 'lrneg'] = param_fits[2]
        M8eps_2lr_uc_lrs.at[i, 'uc'] = param_fits[3]
        M8eps_2lr_uc_lrs.at[i, 'lrs'] = param_fits[4]
        M8eps_2lr_uc_lrs.at[i, 'w'] = param_fits[5]
        M8eps_2lr_uc_lrs.at[i, 'negLL'] = negLL
        M8eps_2lr_uc_lrs.at[i, 'BIC'] = BIC
        M8BIC_all.at[i, 'eps-2lr-uc-lrs'] = BIC
        title_excel = os.path.join(save_dir, f'HaS1var_M8_8modelrecov_eps_2lr_uc_lrs.xlsx')
        M8eps_2lr_uc_lrs.to_excel(title_excel, index=False)

        title_excel = os.path.join(save_dir, f'HaS1var_M8_modelrecov_BIC_all.xlsx')
        M8BIC_all.to_excel(title_excel, index=False) 
        '''  


'''

rows = np.arange(recov_amount)
BIC_matrix = np.zeros((8, 8))

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

M3BIC = M3BIC_all.to_numpy()
M3BICwin = np.zeros_like(M3BIC)
M3_minindices = np.argmin(M3BIC, axis=1)
M3BICwin[rows, M3_minindices] = 1
BIC_matrix[2, :] = np.sum(M3BICwin, axis=0)/recov_amount

M4BIC = M4BIC_all.to_numpy()
M4BICwin = np.zeros_like(M4BIC)
M4_minindices = np.argmin(M4BIC, axis=1)
M4BICwin[rows, M4_minindices] = 1
BIC_matrix[3, :] = np.sum(M4BICwin, axis=0)/recov_amount

M5BIC = M5BIC_all.to_numpy()
M5BICwin = np.zeros_like(M5BIC)
M5_minindices = np.argmin(M5BIC, axis=1)
M5BICwin[rows, M5_minindices] = 1
BIC_matrix[4, :] = np.sum(M5BICwin, axis=0)/recov_amount

M6BIC = M6BIC_all.to_numpy()
M6BICwin = np.zeros_like(M6BIC)
M6_minindices = np.argmin(M6BIC, axis=1)
M6BICwin[rows, M6_minindices] = 1
BIC_matrix[5, :] = np.sum(M6BICwin, axis=0)/recov_amount

M7BIC = M7BIC_all.to_numpy()
M7BICwin = np.zeros_like(M7BIC)
M7_minindices = np.argmin(M7BIC, axis=1)
M7BICwin[rows, M7_minindices] = 1
BIC_matrix[6, :] = np.sum(M7BICwin, axis=0)/recov_amount

M8BIC = M8BIC_all.to_numpy()
M8BICwin = np.zeros_like(M8BIC)
M8_minindices = np.argmin(M8BIC, axis=1)
M8BICwin[rows, M8_minindices] = 1
BIC_matrix[7, :] = np.sum(M8BICwin, axis=0)/recov_amount

simulations = [r'M1: $\epsilon$-$\alpha$', 
               r'M2: $\epsilon$-$\alpha$-$\lambda$',
               r'M3: $\epsilon$-$\alpha$-$\alpha_{S}$',
               r'M4: $\epsilon$-$\alpha$-$\lambda$-$\alpha_{S}$',
               r'M5: $\epsilon$-$\alpha_{+}$-$\alpha_{-}$',
               r'M6: $\epsilon$-$\alpha_{+}$-$\alpha_{-}$-$\lambda$',
               r'M7: $\epsilon$-$\alpha_{+}$-$\alpha_{-}$-$\alpha_{S}$',
               r'M8: $\epsilon$-$\alpha_{+}$-$\alpha_{-}$-$\lambda$-$\alpha_{S}$',
               ]
model_fits =  [r'M1: $\epsilon$-$\alpha$', 
               r'M2: $\epsilon$-$\alpha$-$\lambda$',
               r'M3: $\epsilon$-$\alpha$-$\alpha_{S}$',
               r'M4: $\epsilon$-$\alpha$-$\lambda$-$\alpha_{S}$',
               r'M5: $\epsilon$-$\alpha_{+}$-$\alpha_{-}$',
               r'M6: $\epsilon$-$\alpha_{+}$-$\alpha_{-}$-$\lambda$',
               r'M7: $\epsilon$-$\alpha_{+}$-$\alpha_{-}$-$\alpha_{S}$',
               r'M8: $\epsilon$-$\alpha_{+}$-$\alpha_{-}$-$\lambda$-$\alpha_{S}$',
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

'''