#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 17:08:27 2023

@author: zhangz31

Function 'buildWorldModel' and 'multiThread' were developed based on repo
https://github.com/ajlangdon/multithreadTD

"""
import numpy as np
from numpy import matlib

def microstimulusSingleRwdBasis(states, rewardstate, nmicrostimulus=10, alpha=0.01,
                               gamma=0.95, lambda_=0.95, sigma=0.08, d=0.99, 
                               lr_decay=-5e-2, maxstatelength=np.nan):
    '''
    Function to simulate microstimulus TDRL model.

    Args:
        eventlog (np.array): Event log array with columns [eventindex, eventtime]
        rewardstate (np.array): Array of event indices of rewards
        nmicrostimulus (int): Number of microstimuli each event elicits
        statesize (int): Size of state
        alpha (float): Learning rate
        gamma (float): Temporal discounting parameter
        lambda_ (float): Eligibility trace parameter
        sigma (float): Width of Gaussian function 
        d (float): Decay parameter
        inhibitionlog (np.array): Array with columns [start time, stop time, inhibition level (rpe)]
        maxstatelength (float, optional): Truncate states at maxstatelength from each cue onset. Default is np.nan.

    Returns:
        rpetimeline (np.array): Array of Reward Prediction Errors (RPEs) over time.
        valuetimeline (np.array): Array of value estimations over time.
        eventtimeline (np.array): Array with columns [state index, reward magnitude]
    '''

    # Calculate number of unique stimuli and session end time
    nstimuli = len(np.unique(states)) # excluded the null observation
    nstate = states.size

    # Initialize value, RPE, weight, eligibility trace, microstimulus level and memory trace arrays
    valuetimeline, rpetimeline = np.zeros((nstate, 1)), np.zeros((nstate, 1))
    w, e, x = np.zeros((3, nmicrostimulus, nstimuli))
    y = np.zeros(nstimuli,)

    # For first time bin, initialize stimulus index and number
    tempstimuli = np.zeros(9, )-1
    tempstimuli[states[0]] = 0
    
    # Initialize reward and RPE
    rewarded = states[0] in rewardstate
    reward = 1 if rewarded else 0
    RPE = reward + gamma*valuetimeline[0]
    rpetimeline[0] = RPE

    # Update weights and eligibility traces for first time bin
    for is_ in range(1, nstimuli):
        w[:,is_] += alpha*rpetimeline[0]*e[:,is_]
        e[:,is_] = gamma*lambda_*e[:,is_] + x[:,is_]

    # Iterate calculation for rest time bins
    for i in range(1, nstate):
        vtemp = 0
        tempstimuli[states[i]] = i
        for is_ in range(1, nstimuli):
            if tempstimuli[is_] >= 0 and i-tempstimuli[is_]<nmicrostimulus:
                y[is_, ] = d**(i-tempstimuli[is_])
                x[:,is_] = (1/np.sqrt(2*np.pi))*np.exp(-((y[is_, ]-np.arange(1,nmicrostimulus+1)/nmicrostimulus)**2)/(2*sigma**2))*y[is_, ]

            vtemp += np.dot(w[:,is_].T, x[:,is_])

        valuetimeline[i] = vtemp
        # Update reward and RPE
        rewarded = states[i] in rewardstate
        reward = 1 if rewarded else 0

        RPE = reward + gamma*valuetimeline[i] - valuetimeline[i-1]
        rpetimeline[i] = RPE

        # Update weights and eligibility traces for rest time bins
        lr_list = []
        for is_ in range(1, nstimuli):
            # lr = alpha * np.exp(lr_decay*stimulusnum[is_])
            lr = np.array(alpha) #  * np.exp(lr_decay*stimulusnum[is_])
            lr_list.append(lr.round(3).tolist())
            w[:,is_] += lr*rpetimeline[i]*e[:,is_]
            e[:,is_] = gamma*lambda_*e[:,is_] + x[:,is_]

    return rpetimeline, valuetimeline, w


def multiThread(events, T, O, M, U, taskstates, reward, eta, gamma, elambda, decay, t_eta, stream_start, stream_end, init_w=0):
    """
    vector TD learning (multithread) with transition updates

    :events: vector ntrials x ntimepoints containing codes for task events during each trial
    :T: matrix nstates x nstates P(state_{t+1}|state_{t})
    :O: matrix nstates x nstates x nobs for P(o_{t+1}|state_{t},state_{t+1}) OR nstates x nobs for P(o_{t}|state_{t})
    :M: maxtrix nids x nstates of mapping from states to vector prediction dimension
    :U: matrix nstates x nids of mapping from vector rpe dimension to states (nb we use M.T)
    :taskstates: maxtrix nstates x ntaskstates of mapping for states into (macro) taskstates [0,1]
    :reward: nids x nobs of reward amount per observation for each dim
    :eta: learning rate 0<=eta<=1
    :gamma: discount 0<=discount<=1
    :elambda: eligibility 0<=elambda<=1
    :decay: trial-by-trial decay in weights 0<=decay<=1
    :init_w: initial weights
    """
    # values are vectorized, RPE is also vectorized
    # M controls the dimensionality of the prediction errors
    # i.e. maps from timepoint states to discrete 'channels' of prediction

    # we also have updates of the transition matrix according to observations in a trial

    # here, reward should be nids x nobs

    nstates = T.shape[0] # number of states
    ntr = events.shape[0] # number of trials
    t_max = events.shape[1] # number of timepoints
    nids = M.shape[0] # dimensionality of the prediction
    # check orientation of taskstates
    if taskstates.shape[1]>taskstates.shape[0]:
        taskstates = taskstates.T
    ntstates = taskstates.shape[1] # number of taskstates
    # print(type(taskstates[0,0]))

    # states
    init_state = np.zeros(nstates)
    init_state[0] = 1 # start in background on each trial

    # save states and values
    state = np.zeros([ntr,nstates,t_max])
    nextstate = np.zeros([ntr,nstates,nstates,t_max]) # this is the full progression
    weights = np.zeros([ntr,nstates,t_max])
    eligibility = np.zeros([ntr,nstates,t_max])
    value = np.zeros([ntr,nids,t_max])
    nextvalue = np.zeros([ntr,nids,t_max])
    rpe = np.zeros([ntr,nids,t_max])
    activethread = np.zeros([ntr,ntstates,t_max])

    w = np.zeros(nstates) + init_w # evolving weights

    # and evolving transition matrix
    tmatrix = np.zeros([ntr,nstates,nstates])
    tmatrix[0,:,:] = T

    for tr in range(ntr):

        # initial state is always background pre-cue
        s = init_state
        e = np.zeros(nstates) # evolving eligibility traces *within trial*
        thread = np.zeros(ntstates) # evolving taskstate occupancy

        # outcome tracking for transition learning
        tr_outcome = np.zeros([nids])

        # thread activity
        threadhistory = np.zeros([ntstates])

        for t in range(t_max-1):
            # track next obs
            if events[tr,t+1]>1:
                tr_outcome[events[tr,t+1]-2] += 1

            # extract current value from state representation
            v = M@(w*s) # current value of current state
            # calculate values of next states *within channels*
            smap = np.reshape(s,[nstates,1]) # reshape operates in place so lets reassign
            sproj = matlib.repmat(smap,1,nstates) # projection of current state to right dimensions (i.e. explicit transpose)
            if len(np.shape(O))==2:
                oproj = matlib.repmat(np.reshape(O[:,events[tr,t+1]],[1,nstates]),nstates,1) # the obs P that goes with sprime
            elif len(np.shape(O))==3:
                oproj = O[:,:,events[tr,t+1]] # the obs P that goes with sprime conditional on s
            else:
                print('dim error in O')

            sprime = sproj*(tmatrix[tr,:,:]*oproj)
            sprime = sprime/np.sum(sprime[:]) # normalize
            # compute next state value *within channel*
            vprime = M@sprime@w

            # tracking current thread at the level of taskstates
            # print(s.shape),print(taskstates.shape),print(threadhistory.shape)
            thread = s@taskstates
            # tracking thread history
            threadhistory = np.maximum(threadhistory,thread)

            # first decay eligibility traces of previous states
            e = gamma*elambda*e
            # update eligibility with current state
            e = np.clip(e + s,0,1) # these can't be >1 for any state that we re-enter (e.g. background)
            # for consistency, let's zero out the eligibility of the bg
            e[0] = 0
            # eligibility is limited by thread occupancy
            # note eligibility update was already proportional to belief
            for ts in range(ntstates):
                e[taskstates[:,ts]] = e[taskstates[:,ts]]*(thread[ts]>0)

            # learning for state at t
            # prediction error should be vector of length nids
            # in the appropriate ID channel
            delta = reward[:,events[tr,t+1]] + gamma*vprime - v
            # state weights are updated by these prediction errors according to channel membership
            w = w + eta*e*(U@delta) # update weights in state space
            # but we assume the background does not accrue predictions
            w[0] = 0

            state[tr,:,t] = s # save belief states
            nextstate[tr,:,:,t] = sprime
            weights[tr,:,t] = w # and weights
            eligibility[tr,:,t] = e # and eligibility traces
            value[tr,:,t] = v # timepoint values
            nextvalue[tr,:,t] = vprime
            activethread[tr,:,t] = thread
            rpe[tr,:,t] = delta # and rpe
            
            s = sprime.sum(axis=0) # transition to next state


        # trial-to-trial loss in threads that were active
        w = (1-decay*(taskstates@threadhistory))*w

        # let's update the transition matrix at the end of the trial
        # normalize the observed transitions to make it P
        tr_outcome = tr_outcome/tr_outcome.sum()

        if tr < (ntr-1):
            tmatrix[tr+1,:,:] = T # keep all transitions
            # learn transitions only for the starting states of each thread
            for n in range(nids):
                if tr_outcome[n]==1: # if this transition happened
                    tmatrix[tr+1,0,stream_start[n]] = tmatrix[tr,0,stream_start[n]] + t_eta*(1 - tmatrix[tr,0,stream_start[n]])
                else: # this transition didn't happen
                    tmatrix[tr+1,0,stream_start[n]] = tmatrix[tr,0,stream_start[n]]*(1-t_eta)
                    
    return state, nextstate, weights, eligibility, value, nextvalue, rpe, activethread, tmatrix



def buildWorldModel(chains):
    """
    For odor-guided identity change task (Takahashi et al, 2023)
    """
    # observations 
    nobs = 4 # 0 = null, 1 = cue, 2 = chocolate, 3 = vanilla

    # will assume all states are contiguous
    states = np.concatenate(chains)
    nstates = len(states)
    taskstates = np.arange(len(chains))
    ntaskstates = len(taskstates)

    bg = chains[0][0] # chain[0] is the bg and is only 1 state

    # transition matrix
    T = np.zeros([nstates,nstates])
    # bg -> bg OR first state in chain 1,2,3
    T[0,0] = 0.5 # background to self
    T[0,[chains[1][0],chains[2][0]]] = (1-0.5)/2
    # chain 1 progression within chain
    T[chains[1][0:-1],chains[1][1:]] = 1
    # # chain 1 transition to bg
    # T[chains[1][0:-1],bg] = 0
    # final state in chain 1 transition to bg
    T[chains[1][-1], bg] = 1
    # chain 2 progression within chain
    T[chains[2][0:-1],chains[2][1:]] = 1
    # # or to background
    # T[chains[2][0:-1],bg] = 0
    # final state in chain to bg
    T[chains[2][-1],bg] = 1

    # 3D observation matrix for transition specific observation probabilities
    O = np.zeros([nstates,nstates,nobs])

    ou_self = 0.5 # P(null| self->self transition)
    # this has to be the same for all self-> self transitions under the null observation
    # *with* same P(self->self) to avoid dynamic readjustment of the state occupancy from
    # the normalization

    # null == bg->bg
    O[bg,bg,0] = 1
    O[bg,bg,2] = (1-ou_self)/2
    O[bg,bg,3] = (1-ou_self)/2

    # cue == bg->start of each stream
    O[bg,[chains[1][0],chains[2][0]],1] = 1/2

    # null, reward == within chain 1,2 progression
    O[chains[1][0]:chains[1][-1], chains[1][1]:(chains[1][-1]+1),0] = np.identity(len(chains[1])-1)*ou_self
    O[chains[1][0]:chains[1][-1], chains[1][1]:(chains[1][-1]+1),2] = np.identity(len(chains[1])-1)*ou_self
    O[chains[2][0]:chains[2][-1], chains[2][1]:(chains[2][-1]+1),0] = np.identity(len(chains[2])-1)*ou_self
    O[chains[2][0]:chains[2][-1], chains[2][1]:(chains[2][-1]+1),3] = np.identity(len(chains[2])-1)*ou_self
    # last state of each chain transitions to bg under null
    O[chains[1][-1],bg,0] = 1
    O[chains[2][-1],bg,0] = 1

    # now the reward observations control the reset behavior within stream
    # single stream reset for each of these rewards
    # null == stream 1 -> bg
    O[chains[1][0]:chains[1][-1],bg,2] = 1
    # null == stream 2 -> bg
    O[chains[2][0]:chains[2][-1],bg,3] = 1

    # other reward observations allow progression
    # choc allows continuation of streams 2
    O[chains[2][0]:chains[2][-1],chains[2][1]:(chains[2][-1]+1),2] = np.identity(len(chains[2])-1)
    # vanilla allows continuation of stream 1
    O[chains[1][0]:chains[1][-1],chains[1][1]:(chains[1][-1]+1),3] = np.identity(len(chains[1])-1)

    # taskstates group the chains
    taskstates = np.zeros([nstates,ntaskstates],dtype='bool') # must be boolean for the indexing I use
    for c in range(ntaskstates):
        taskstates[chains[c],c] = True

    return T, O, taskstates


