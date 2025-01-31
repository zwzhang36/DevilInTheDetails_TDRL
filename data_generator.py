#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 15:53:35 2023

@author: zhangz31
"""

# generate data sequence

import copy
import numpy as np

from itertools import chain
from scipy.stats import expon
from numpy.random import choice, shuffle

# In[] For trial-less cue-reward task

def sampleTrialLess(n_trials, trialLess=False):
    """
    Generate trial-less task observations.
    Observation: 0: null; 1: cue; 2: reward
    """
    # Parameters 
    mean_interval = 30  # Mean interval between cues
    min_interval = 2    # Minimum possible interval
    max_interval = 90   # Maximum possible interval
    
    # Define exponential distribution with the specified mean interval
    scale = mean_interval  # Scale parameter for the exponential distribution
    rv = expon(scale=scale)
    
    # Generate range of possible intervals
    x = np.linspace(min_interval, max_interval, max_interval - min_interval + 1)
    
    # Compute the PDF and normalize for truncated distribution
    pdf = rv.pdf(x)
    pdf[5] = 0  # The previous reward and intermediate cue cannot present at the same time
    pdf[0] = 0  # Preventing extremely short intervals
    if not trialLess:
        pdf[:10] = 0  # Stricting short intervals if trial-based
    normalized_pdf = pdf / pdf.sum()  # Normalize
    
    # Observation matrix
    # States: 0 = null, 1 = cue on, 2 = reward
    o_list = []
    
    # Generate trials in chunks of 1000 for efficiency
    for i in range(np.ceil(n_trials / 1000).astype(int)):
        num = 1000 if i < int(n_trials / 1000) else n_trials - 1000 * i
        
        while True:
            # Sample cue intervals based on the PDF
            interval_cue = choice(range(len(normalized_pdf)), num, p=normalized_pdf)
            interval_cue = np.cumsum(interval_cue)  # Convert to cumulative time points
            
            # Ensure no overlapping between cue and reward
            if np.intersect1d(interval_cue, interval_cue + 5).size == 0:
                break
        
        # Initialize observation array with zeros (no events by default)
        observations = np.zeros(interval_cue[-1] + 10, dtype=int)
        
        # Assign cue onset (1) and reward (2) at respective positions
        observations[interval_cue] = 1
        observations[interval_cue + 5] = 2  # Reward occurs 5 seconds after cue
        
        # Convert observations to a list and append to the main list
        o_list.extend(observations.tolist())
    
    return o_list

# In[] For contingency degrdation
    
# Parameters for dwell time
dt = 0.1  # Time step in seconds
t_range = 25  # Total time range in seconds

# Transition matrix (T) initialization
T = np.zeros((3, 3))  # 3 states: cue1, cue2, ITI (inter-trial interval)
T[0, 2] = 1  # Transition from cue 1 to ITI
T[1, 2] = 1  # Transition from cue 2 to ITI

# Copy transition matrices for different trial types
T_12, T_34 = copy.deepcopy(T), copy.deepcopy(T)
T_12[2, 0] = 1/12  # Transition from ITI to cue 1
T_12[2, 2] = 11/12  # Remaining in ITI
T_34[2, 1] = 1/12  # Transition from ITI to cue 2
T_34[2, 2] = 11/12  # Remaining in ITI

# Function to assign dwell distribution
def assign(t_event):
    temp = np.zeros(int(t_range / dt))
    if isinstance(t_event, (float, int)):
        temp[round(t_event / dt)] = 1
    elif isinstance(t_event, list):
        for i in t_event:
            temp[round(i / dt)] = 1 / len(t_event)
    return temp

# Function to sample continuous degraded trials
def sampleContDeg(n_trials, degraded=False, state_space='partial'):
    trial_type = [2, 4] if degraded else [1, 3]
    trial_type = trial_type * (n_trials // 2)
    shuffle(trial_type)
    
    observations = [[0]]  # Initialize observations
    
    # Define observation matrices for different transitions
    O = np.zeros((3, 3, 8))  # Observation matrix 
    O[0, 2, 0] = 0.5  # Cue 1 -> ITI -> Cue 1 off
    O[0, 2, 3] = 0.5  # Cue 1 -> ITI -> Cue 1 off and reward 1
    O[1, 2, 0] = 0.5  # Cue 2 -> ITI -> Cue 2 off
    O[1, 2, 6] = 0.5  # Cue 2 -> ITI -> Cue 2 off and reward 2
    O[2, 0, 1] = 1  # ITI -> Cue 1 -> Cue 1 on
    O[2, 1, 4] = 1  # ITI -> Cue 2 -> Cue 2 on
    
    O_13, O_24 = copy.deepcopy(O), copy.deepcopy(O)
    O_13[2, 2, 0] = 1  # ITI -> ITI -> None
    O_24[2, 2, 0] = 0.5  # ITI -> ITI -> None
    O_24[2, 2, 7] = 0.5  # ITI -> ITI -> Reward 1
    
    D = {0: assign(0.5), 1: assign(0.5), 2: assign(0.5)}  # Dwell distribution
    
    info_Matrix = [[T_12, O_13, D], 
                   [T_12, O_24, D],
                   [T_34, O_13, D],
                   [T_34, O_24, D]]
    
    # Function to compute state duration probabilities
    def func(n):
        return 1.2 ** n
    
    temp = np.array(list(map(func, range(6, 0, -1))))
    temp = temp / temp.sum()
    
    p_temp = copy.deepcopy(info_Matrix[trial_type[0]-1][2][0])
    p_temp[:] = 0
    p_temp[:6] = temp
    
    for n in range(n_trials):
        T, O, D = info_Matrix[trial_type[n] - 1]
        if trial_type[n] in [1, 2]:
            state, observation = [0], [1]  # Cue 1
        else:
            state, observation = [1], [4]  # Cue 2
        
        while True:
            prob_s = T[state[-1], :].reshape(-1, )
            new_state = choice(range(T.shape[0]), 1, p=prob_s)[0]
            if new_state in [0, 1, 3] or len(state) > 43:
                break
            state.append(new_state)
            
            prob_o = O[state[-2], state[-1], :].reshape(-1, )
            observation.append(choice(range(O.shape[2]), 1, p=prob_o)[0])
        
        for ith, (s, o) in enumerate(zip(state, observation)):
            duration = choice(range(D[s].shape[0]), 1, p=D[s])[0]
            if ith == len(state) - 1:
                observations.append([0] * duration)
                duration = choice(range(D[s].shape[0]), 1, p=p_temp)[0]
                observations.append([0] + [0] * (duration - 1))
            else:
                observations.append([o] + [0] * (duration - 1))
    
    return list(chain(*observations))


# In[] For odor-guided identity change task

def sampleOdorGuidedTask(dt, t_max, ntr, blocktr):
    """
    Generate task events
    """

    t_cue = int(0.5/dt)
    # cue == 1, terminal reward == 2, chocolate == 3, vanilla == 4

    well1 = np.zeros([ntr,t_max],dtype='int')
    well1[:,t_cue] = 1
    well2 = np.copy(well1)
    
    #------ well1
    # block 1 - big chocolate
    well1[:blocktr,t_cue+int(0.5/dt)] = 2
    well1[:blocktr,t_cue+int(1.0/dt)] = 2
    well1[:blocktr,t_cue+int(1.5/dt)] = 2
    # block 2 - small chocolate
    well1[blocktr:blocktr*2,t_cue+int(0.5/dt)] = 2
    # block 3 - small vanilla
    well1[(blocktr*2):blocktr*3,t_cue+int(0.5/dt)] = 3
    # block 4 - big vanilla
    well1[(blocktr*3):blocktr*4,t_cue+int(0.5/dt)] = 3
    well1[(blocktr*3):blocktr*4,t_cue+int(1.0/dt)] = 3
    well1[(blocktr*3):blocktr*4,t_cue+int(1.5/dt)] = 3
    # block 5 - big chocolate
    well1[blocktr*4:,t_cue+int(0.5/dt)] = 2
    well1[blocktr*4:,t_cue+int(1.0/dt)] = 2
    well1[blocktr*4:,t_cue+int(1.5/dt)] = 2
    
    #------ well2
    # block 1 - small vanilla
    well2[:blocktr,t_cue+int(0.5/dt)] = 3
    # block 2 - big vanilla
    well2[blocktr:blocktr*2,t_cue+int(0.5/dt)] = 3
    well2[blocktr:blocktr*2,t_cue+int(1.0/dt)] = 3
    well2[blocktr:blocktr*2,t_cue+int(1.5/dt)] = 3
    # block 3 - big chocolate
    well2[(blocktr*2):blocktr*3,t_cue+int(0.5/dt)] = 2
    well2[(blocktr*2):blocktr*3,t_cue+int(1.0/dt)] = 2
    well2[(blocktr*2):blocktr*3,t_cue+int(1.5/dt)] = 2
    # block 4 - small chocolate
    well2[blocktr*3:blocktr*4,t_cue+int(0.5/dt)] = 2
    # block 5 - small vanilla
    well2[(blocktr*4):,t_cue+int(0.5/dt)] = 3

    return well1, well2

