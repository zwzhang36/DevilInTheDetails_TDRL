#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 09:33:48 2025

@author: zhangz31
"""


import numpy as np
import matplotlib.pyplot as plt

from models import microstimulusSingleRwdBasis
from data_generator import sampleTrialLess

# In[0] Trial less task

# Number of trials
n_trials = 5000

# Dictionary to store RPEs for different event types
rpes = {'cue1': [], 'cue2': [], 'rwd1': [], 'rwd2': []}

# Repeat the simulation for robustness
for nrepeat in range(20):
    # Generate observations
    #    Pre-training without intermediate cues can improve alignment with 
    #    dopamine responses, but this does not change results qualitatively.
    # Observation: 0: null; 1: cue; 2: reward
    obs_trial = np.array(sampleTrialLess(n_trials, trialLess=False))  # no intermediate observations
    obs_trialless = np.array(sampleTrialLess(n_trials * 2, trialLess=True))  # Trial-less observations
    
    # Concatenate observations
    obs = np.concatenate((obs_trial, obs_trialless))
    
    # Identify indices where cue appears (cue coded as 1)
    ind_cue = np.where(obs == 1)[0]
    
    # Find intermediate cues
    intermediateidx = ind_cue[1:][np.logical_and(np.diff(ind_cue) < 5, np.diff(ind_cue) >= 2)]
    
    # Mark intermediate cues as state 3 directly for simplify
    states = obs.copy()
    states[intermediateidx] = 3
    
    # Simulate using microstimulus learning model
    rpe, value_timeline, value = microstimulusSingleRwdBasis(states, 
                                                             rewardstate=[2],  # Reward state is coded as 2
                                                             d=0.9,  # Discount factor
                                                             lr_decay=1e-3,  # Learning rate decay
                                                             nmicrostimulus=10)  # Number of microstimuli
    
    # Store RPEs
    rpes['cue1'].append(rpe[states == 1][n_trials:, :])  # Initial cue
    rpes['cue2'].append(rpe[states == 3][:, :])          # Intermediate cue
    rpes['rwd1'].append(rpe[np.where(states == 1)[0][n_trials:] + 5])  # Isnitial reward 
    rpes['rwd2'].append(rpe[np.where(states == 3)[0][:] + 5])          # Intermediate reward 

# Plot ratios and save
def plot_ratio(data1, data2, ylabel, filename, ylim):
    plt.figure(figsize=(3, 5))
    ratio = np.array([data2[i].mean() / data1[i].mean() for i in range(len(data1))])
    plt.bar([1], ratio.mean(), yerr=ratio.std() / np.sqrt(ratio.size), width=0.25)
    plt.hlines(1, 0.8, 1.2, color='k', linestyle='--')
    plt.xticks([1], [''])
    plt.xlim([0.8, 1.2])
    plt.ylim([0, ylim])
    plt.ylabel(ylabel, fontsize=16)
    plt.savefig(f'{filename}.eps', bbox_inches='tight')
    plt.savefig(f'{filename}.png', bbox_inches='tight')

# Plot cue response ratio
plot_ratio(rpes['cue1'], rpes['cue2'], 
           'Ratio of Cue Response \n (Intermediate / Previous)', 
           'ratio_of_cue_response', 1.25)

# Plot reward response ratio
plot_ratio(rpes['rwd1'], rpes['rwd2'], 
           'Ratio of Reward Response \n (Intermediate / Previous)',
           'ratio_of_rwd_response', 3.5)
