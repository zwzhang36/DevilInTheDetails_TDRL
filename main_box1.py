#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 09:34:49 2025

@author: zhangz31

Contingency degradation task

"""

import numpy as np
import matplotlib.pyplot as plt

from models import microstimulusSingleRwdBasis
from data_generator import sampleContDeg

# Observations: 0: a null observation; 
#               1: cue 1 on;  2:cue 1 off; 3: cue 1 off and reward 1; 
#               4: cue 2 on;  5:cue 2 off; 6: cue 2 off and reward 2;  
#               7: reward 1 during ITI

def rpe2firing(x1, x2=None, pos=1, neg=1):
    """
    Convert RPEs into firing rates.

    """
    def process_rpe(x):
        y = np.array([x])
        y[y < 0] *= neg  # Apply negative scaling to negative values
        y[y > 0] *= pos  # Apply positive scaling to positive values
        return y
    
    if x2 is None:
        return process_rpe(x1)
    else:
        return process_rpe(x1) + process_rpe(x2)  # Combine both RPE inputs

def simulate(d, sigma, gamma, lambda_, nmicrostimulus):
    """
    Simulate the RPE dynamics in the contingency degradation task.
    
    """
    n_trials = 500
    rpes = {key: [] for key in ['cue1', 'cue2', 'rw1_d', 'rw2_d', 'ITI_d_ITI']}
    
    for nrepeat in range(20):
        # Generate observations for acquisition and degraded trials
        obs_acquire = np.array(sampleContDeg(n_trials, degraded=False))
        obs_degraded = np.array(sampleContDeg(n_trials, degraded=True))
        obs = np.concatenate((obs_acquire, obs_degraded))
        
        # Compute RPEs for different reward states
        rpe_1, _, _ = microstimulusSingleRwdBasis(obs, rewardstate=[3, 7], d=d, 
                                                  sigma=sigma, gamma=gamma, 
                                                  lambda_=lambda_,
                                                  nmicrostimulus=nmicrostimulus)
        rpe_2, _, _ = microstimulusSingleRwdBasis(obs, rewardstate=[6], d=d,
                                                  sigma=sigma, gamma=gamma, 
                                                  lambda_=lambda_,
                                                  nmicrostimulus=nmicrostimulus)
        
        # Extract degraded RPEs
        rpe_1 = rpe_1[-obs_degraded.size:, :]
        rpe_2 = rpe_2[-obs_degraded.size:, :]

        def slice_data(x, d, y=obs_degraded, indices=None, length=0):
            """Extract slices of RPE data for given event type d."""
            if length == 0:
                if indices is None:
                    indices = np.where(y == d)[0]
                return np.array([x[i] for i in indices])    
            else:
                if indices is None:
                    indices = np.where(y == d)[0][1:-1]
                return np.array([x[max(0, i - length): min(len(x), i + length)] for i in indices]).squeeze(axis=2)
        
        # Store RPE slices in dictionary
        rpes['cue1'].append([slice_data(rpe_1, 1), slice_data(rpe_2, 1)])
        rpes['cue2'].append([slice_data(rpe_1, 4), slice_data(rpe_2, 4)])
        rpes['rw2_d'].append([slice_data(rpe_1, 6), slice_data(rpe_2, 6)])
        
        # Identify reward-related events
        index = {
            'rw1_d': np.intersect1d(np.where(obs_degraded == 3)[0], np.where(obs_degraded == 1)[0] + 5),
            'ITI_d_ITI': np.setdiff1d(np.where(obs_degraded == 3)[0], np.where(obs_degraded == 1)[0] + 5)
        }
        
        # Store RPEs for reward-related events
        for key, values in index.items():
            rpes[key].append([rpe_1[values], rpe_2[values]])
    
    return rpes

def plot_rpes(rpes, save=True):
    """
    Plot RPE data.
    
    """
    rpes_1, rpes_2, rpes_all = {}, {}, {}
    
    # Group RPE data
    for key in ['cue1', 'cue2', 'rw1_d', 'rw2_d']:
        rpes_1[key] = [np.mean(rpe2firing(i[0], pos=1, neg=1), axis=1) for i in rpes[key]]
        rpes_2[key] = [np.mean(rpe2firing(i[1], pos=1, neg=1), axis=1) for i in rpes[key]]
        rpes_all[key] = [np.mean(rpe2firing(i[0], i[1], pos=10, neg=4), axis=1) for i in rpes[key]]
    
    label_titles = ['rwd basis 1', 'rwd basis 2', 'rwd basis all']
    label_suptitles = ['cue', 'rwd', 'rwd 1 vs ITI']
    
    # Generate box plots 
    for ith, (key1, key2) in enumerate(zip(['cue1', 'rw1_d'], ['cue2', 'rw2_d'])):
        plt.figure(figsize=(8, 5))
        
        for jth, rpes_ in enumerate([rpes_1, rpes_2, rpes_all]):
            plt.subplot(1, 3, jth + 1)
            x1, x2 = np.array(rpes_[key1]).squeeze(), np.array(rpes_[key2]).squeeze()
            plt.boxplot([x1, x2], widths=0.3)
            plt.hlines(0, 0.5, 2.5, linestyles='dashed', color='k')
            plt.xticks([1, 2], [key1, key2])
            plt.title(label_titles[jth])
        
        plt.suptitle(label_suptitles[ith])

d = 0.9
sigma = 0.25
gamma   = 0.95
lambda_ = 0.8
nmicrostimulus = 10

# microstimulus
rpes = simulate(d, sigma, gamma, lambda_, nmicrostimulus)
plot_rpes(rpes)
plt.show()


