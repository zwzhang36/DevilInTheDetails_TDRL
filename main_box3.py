#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:20:53 2023

@author: zhangz31

Identify shift tasks

"""

import numpy as np
import matplotlib.pyplot as plt

from models import multiThread, buildWorldModel
from data_generator import sampleOdorGuidedTask


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


# Time step and state definitions
dt = 0.1  # Time step in seconds
d_max = int(2.0 / dt)  # Maximum duration in discrete steps
t_max = int(2.5 / dt)  # Total task duration in discrete steps

n_sessions = 5
# Task and block definitions
blocktr = 50  # Trials per block
nblock = 5  # Number of blocks
ntr = blocktr * nblock  # Total number of trials

# Generate task trials (odor-guided task simulation)
# well1 and well2 correspond to two different reward wells
well1, well2 = sampleOdorGuidedTask(dt, t_max, ntr, blocktr)
well1 = np.tile(well1, (n_sessions, 1))
well2 = np.tile(well2, (n_sessions, 1))

# Define state and observation parameters
nstates = int(d_max) + 1  # Number of states including ITI (Inter-Trial Interval)
nids = 2  # Number of reward identities (chocolate, vanilla)

# Define transition threads
threads = [[0], np.arange(1, d_max + 1), np.arange(d_max + 1, d_max * 2 + 1)]

# Build the transition and observation model
T, O, taskstates = buildWorldModel(threads)

# Mapping states to reward prediction space
M = np.zeros([nids, T.shape[0]])
M[:, 0] = 1 / 2  # Initial uniform probability
M[0, threads[1]] = 1  # Map chocolate-associated states
M[1, threads[2]] = 1  # Map vanilla-associated states

# Define reward matrix mapping observations to reward amounts
reward = np.zeros([nids, O.shape[2]])  # Initialize reward matrix
reward[0, 2] = 1  # Chocolate reward
reward[1, 3] = 1  # Vanilla reward

# Learning parameters
eta = 0.3  # Learning rate
gamma = 0.95  # Discount factor
elambda = 1  # Eligibility trace decay
decay = 0.1  # Decay factor for eligibility trace

t_eta = 0.6  # Transition learning rate

# Indices of initial states in transition learning
stream_start = np.array([threads[1][0], threads[2][0]], dtype=int)
stream_end = np.array([threads[1][-1], threads[2][-1]], dtype=int)

# Store Reward Prediction Errors (RPEs)
rpes = {'fst': [], 'lst': []}

# Run multi-threaded simulations for 5 iterations
for i in range(20):
    # Simulate task execution for both wells
    state1, nextstate1, w1, elig1, v1, nextv1, rpe1, thread1, tmatrix1 = multiThread(
        well1, T, O, M, M.T, taskstates, reward, eta, gamma, elambda, decay, t_eta, stream_start, stream_end)
    state2, nextstate2, w2, elig2, v2, nextv2, rpe2, thread2, tmatrix2 = multiThread(
        well2, T, O, M, M.T, taskstates, reward, eta, gamma, elambda, decay, t_eta, stream_start, stream_end)
    
    # Only keep the last seesion for further analysis
    rpe1 = rpe1[-blocktr*nblock:, :, :]
    rpe2 = rpe2[-blocktr*nblock:, :, :]
    
    # Extract and store RPE values
    # First reward drops in different blocks
    rpes['fst'].append(rpe1[blocktr * 2 - 3: blocktr * 2 + 5, :, 9])
    rpes['fst'].append(rpe1[blocktr * 4 - 3: blocktr * 4 + 5, :, 9])
    rpes['fst'].append(rpe2[blocktr * 2 - 3: blocktr * 2 + 5, :, 9])
    rpes['fst'].append(rpe2[blocktr * 4 - 3: blocktr * 4 + 5, :, 9])
    
    # Last reward drops in different blocks
    rpes['lst'].append(rpe1[blocktr * 3 - 3: blocktr * 3, :, 9])
    rpes['lst'].append(rpe1[blocktr * 5 - 3:, :, 9])
    rpes['lst'].append(rpe2[blocktr * 3 - 3: blocktr * 3, :, 9])
    rpes['lst'].append(rpe2[blocktr * 5 - 3:, :, 9])


rpes['fst'] = rpe2firing(np.array(rpes['fst']), pos=5, neg=2).sum(axis=0).sum(axis=2)
rpes['lst'] = rpe2firing(np.array(rpes['lst']), pos=5, neg=2).sum(axis=0).sum(axis=2)


plt.figure()
plt.errorbar(range(3), rpes['fst'][:, :3].mean(axis=0), 
             yerr = rpes['fst'][:, :3].std(axis=0)/np.sqrt(rpes['fst'].shape[0]),
             color = 'k')

plt.errorbar(range(4, 9), rpes['fst'][:, 3:8].mean(axis=0), 
             yerr = rpes['fst'][:, 3:8].std(axis=0)/np.sqrt(rpes['fst'].shape[0]),
             color = 'k')
plt.errorbar(range(10, 13), rpes['lst'].mean(axis=0), 
             yerr = rpes['lst'].std(axis=0)/np.sqrt(rpes['lst'].shape[0]),
             color = 'k')

plt.vlines(3, 0, 4.5, linestyle='dashed', color='k')
plt.xticks([1, 3, 6, 11], ['prev 3', 'shift', 'early 10', 'last 3'])
plt.ylabel('RPE', fontsize=16)
plt.show()
