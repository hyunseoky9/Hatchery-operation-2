# 3.3.7 env analyzer
from Hatchery3_3_7 import Hatchery3_3_7
from scipy.stats import poisson
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import clear_output
def simulator_hatchery3_3_7(envobsvars):
    episodenum = 1
    runtime = 100
    numsteps = []
    Ne_scores = []
    rewards = []
    #env = Hatchery3_3_4(None,1,-1,1,1,{'c':5,'no_genetics':0}) # initstate,parameterizationID,discretization,LCpredmethod
    #envobsvars = ['logcatch_r_apr','logcatch_r_oct','logcatch_r_nov','logcatch_r_apr+nov','logcatch_r_apr+oct+nov','logcatch_r_sep','logcatch_r_aug+sep','logcatch_r_jul+aug+sep','logmaxcatch_r_apr','logmaxcatch_r_oct',
    #            'logmaxcatch_r_nov','effort_r_apr','effort_r_oct','effort_r_nov','effort_r_apr+nov','effort_r_apr+oct+nov','effort_r_jul+aug+sep','effort_r_aug+sep','effort_r_sep','logcpue_r_apr','logcpue_r_sep',
    #            'logcpue_r_nov','logcpue_r_apr+nov','logcpue_r_apr+oct+nov','logcpue_r_jul+aug+sep','numsamples_r_apr','numsamples_r_jul','numsamples_r_aug+sep','numsamples_r_sep','numsamples_r_oct',
    #            'numsamples_r_nov','numsamples_r_jul+aug+sep','numsamples_r_apr+may+jun+jul+aug+sep','prop0_r_apr','prop0_r_oct',
    #            'prop0_r_nov','prop0_r_apr+nov','poolprop_r_apr','poolprop_r_oct','poolprop_r_nov','poolprop_r_apr+nov','logecatch_r_apr','logecatch_r_oct',
    #            'logecatch_r_nov','logecatch_r_apr+nov','logecatch_r_apr+oct+nov','logecatch_r_jul+aug+sep','logecatch_r_sep','logecatch_r_aug+sep']
#envobsvars = ['logcatch_r_apr','logcatch_r_oct','logcatch_r_nov','logcatch_r_apr+nov','numsamples_r_apr','numsamples_r_oct','numsamples_r_nov','effort_r_apr','effort_r_oct','effort_r_nov']
    env = Hatchery3_3_7(None,1,-1,1,1,{'c':5,'no_genetics':0,'obsvars': envobsvars}) # initstate,parameterizationID,discretization,LCpredmethod
    #env = Hatchery3_3_1(None,1,-1,1,1,{'c':5,'no_genetics':0}) # initstate,parameterizationID,discretization,LCpredmethod
    #env = Hatchery3_4_1(None,1,-1,1,1,{'c':5,'no_genetics':0}) # initstate,parameterizationID,discretization,LCpredmethod
    #env = Hatchery3_3_2(None,1,-1,1,1,{'c':5,'no_genetics':0}) # initstate,parameterizationID,discretization,LCpredmethod
    #env = Hatchery3_3_2_2(None,1,-1,1,1,{'c':5,'no_genetics':0}) # initstate,parameterizationID,discretization,LCpredmethod
    #env = Hatchery3_3_3(None,1,-1,1,1,{'c':0,'no_genetics':0}) # initstate,parameterizationID,discretization,LCpredmethod
    for k in range(episodenum):
        env.reset()
        print(f'sz: {env.sz}')
        #with pd.option_context('display.max_rows', None):
        #    print(env.mdata)
        states = [env.state]
        obs = [env.obs]
        mdatas = [env.mdata.copy()]
        done = False
        score = 0
        extinct_period = -1
        srates = [0]
        extinct_recorded = 0
        i = 0
        while done == False:
            if i == 526:
                foo = 0
            action = np.array([1,0.333,0.333,0.333]) 
            #clear_output(wait=True)  # Clear previous output
            #print(f'step: {i}')
            season = 'fall' if env.state[env.sidx['t'][0]] % 2 == 1 else 'spring'
            nextseason = 'spring' if season == 'fall' else 'fall'
            #print(f'season: {season}\n N0total: {np.sum(np.exp(env.state[env.sidx["logN0"]])-1)}\n N1total: {np.sum(np.exp(env.state[env.sidx["logN1"]])-1)}')
            #print(f'N0: {np.exp(env.state[env.sidx["logN0"]])-1}')
            #print(f'N1: {np.exp(env.state[env.sidx["logN1"]])-1}')
            foo = env.step(action)
            #print(f'season: {nextseason}\n N0total: {np.sum(np.exp(env.state[env.sidx["logN0"]])-1)}\n N1total: {np.sum(np.exp(env.state[env.sidx["logN1"]])-1)}')
            #print(f'N0: {np.exp(env.state[env.sidx["logN0"]])-1}')
            #print(f'N1: {np.exp(env.state[env.sidx["logN1"]])-1}')
            #print(f'monitoring data in {nextseason}')
            #with pd.option_context('display.max_rows', None, 
            #                    'display.max_columns', None, 
            #                    'display.width', 2000,
            #                    'display.max_colwidth', None):
            #    print(env.mdata)
            #print(env.state)
            score+=foo[1]
            #print(env.state)
            done = foo[2]
            states.append(env.state)
            obs.append(env.obs)
            mdatas.append(env.mdata.copy())
            #input('pause')
            if i >= runtime:
                done = True # stop after 100 steps
            i += 1
        extinct_period = i - 1
        numsteps.append(extinct_period)
        #print(f'percent change in G: {percentchange_in_G}, initial G: {states[0][env.sidx["G"][0]]}, final G: {states[-1][env.sidx["G"][0]]}')
        #print(f'extinct period: {extinct_period} ({np.floor(extinct_period/4)} years)')
        rewards.append(score)
    obs = np.array(obs)
    states = np.array(states)
    return obs,states