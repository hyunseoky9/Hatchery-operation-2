# run the simulation
import os 
import torch
from Hatchery3_2_2 import Hatchery3_2_2
from scipy.stats import poisson
import random
import pandas as pd
from stacking2 import *
import numpy as np
import pickle
# Set the seed
seednum = np.random.randint(0, 1000000) #389347 #np.random.randint(0, 1000000) #456971 #np.random.randint(0, 1000000) #161384  #798152
random.seed(seednum)
np.random.seed(seednum)
print(seednum)
episodenum = 1
runtime = 50
numsteps = []
extinct_season = []
Ls = []
summer_mort = []
Gpchange = [] # percent change in G
wild_rel2_maxcap = []
Ne_scores = []
rewards = []
# set up the policy network and RMS
config = {'method':0,'seed':943778,'paramset':4} # 68413
print(f'method: TD3, seed: {config['seed']}, paramset: {config['paramset']}')
env = Hatchery3_2_2(None,1,-1,1) # initstate,parameterizationID,discretization,LCpredmethod
device = torch.device('cpu')  # Force CPU usage
wd = f'./TD3 results/good_ones/seed{config['seed']}_paramset{config['paramset']}'
method_str = 'TD3'
filename= f"{wd}/bestPolicyNetwork_{env.envID}_par{env.parset}_dis{env.discset}_TD3.pt"
rmsfilename = f"{wd}/rms_{env.envID}_par{env.parset}_dis{env.discset}_TD3.pkl"
if os.path.exists(rmsfilename):
    standardize = True
    with open(rmsfilename, "rb") as f:
        rms = pickle.load(f)
Policy = torch.load(filename, weights_only=False)
Policy = Policy.to(device) 
fstack = 1 if not hasattr(Policy, 'fstack') else Policy.fstack


for k in range(episodenum):
    init = -1*np.ones(8)
    init[-1] = 0
    env = Hatchery3_2_2(init,1,-1,1) # initstate,parameterizationID,discretization,LCpredmethod
    states = [env.state]
    obs = [env.obs[0]]
    done = False
    score = 0
    extinct_period = -1
    srates = [0]
    extinct_recorded = 0
    newstate = rms.normalize(env.obs) if standardize else env.obs
    stack = np.concatenate([newstate] * fstack)
    i = 0
    while done == False:
        # policy network action
        with torch.no_grad():
            state = torch.tensor(stack, dtype=torch.float32, device=device).unsqueeze(0)  # Add batch dimension
            action = Policy(state).cpu().numpy().flatten()
        newstate  = rms.normalize(env.obs) if standardize else env.obs
        stack = stacking(env,stack,newstate)
        # custom action
        #action = np.array([0.33,0.34,0.33,0]) 
        wild_rel2_maxcap.append(np.sum(np.exp(np.array(env.state)[env.sidx['logN0']])+np.exp(np.array(env.state)[env.sidx['logN1']])-2)/60000)
        foo = env.step(action)
        if foo[2] == False:
            Ls.append(foo[3]['L'])
            summer_mort.append(foo[3]['summer_mortality'])
            Ne_scores.append(foo[3]['Ne_score'])
        else:
            Ls.append(np.ones(3)*(-1))
            Ls.append(np.ones(3)*(-1))
        score+=foo[1]
        #print(env.state)
        done = foo[2]
        states.append(env.state)
        obs.append(env.obs)
        if i < runtime and done == True:
            extinct_season.append(env.obs[-1])
        if i >= runtime:
            done = True # stop after 100 steps
        i += 1
    extinct_period = i - 1
    numsteps.append(extinct_period)
    #print(f'percent change in G: {percentchange_in_G}, initial G: {states[0][env.sidx["G"][0]]}, final G: {states[-1][env.sidx["G"][0]]}')
    if k % 100 == 0:
        print(f'episode {k}')
    #print(f'extinct period: {extinct_period} ({np.floor(extinct_period/4)} years)')
    rewards.append(score)
if episodenum == 1:
    print(f'extinct period: {extinct_period} ({np.floor(extinct_period)} years)')

print(f'average reward: {np.mean(rewards)}, sd: {np.std(rewards)}')
# Convert the list of states to a DataFrame and save to CSV
df_states = pd.DataFrame(states)
# include a column in df_states giving index of the NH state
# total population, total population by reach, proportion of population by age.
df_stats = pd.DataFrame()
if env.discset == -1:
    df_stats['totN'] = np.exp(df_states.iloc[:,0:6]).sum(axis=1) #.sum(axis=0)
    df_stats['totAngo'] = np.exp(df_states.iloc[:,0]) + np.exp(df_states.iloc[:,3])
    df_stats['totIsl'] = np.exp(df_states.iloc[:,1]) + np.exp(df_states.iloc[:,4]) 
    df_stats['totSA'] = np.exp(df_states.iloc[:,2]) + np.exp(df_states.iloc[:,5]) 
    df_stats['totN0'] = np.exp(df_states.iloc[:,0:3]).sum(axis=1) #.sum(axis=0)
    df_stats['totN1'] = np.exp(df_states.iloc[:,3:6]).sum(axis=1) #.sum(axis=0)
    df_stats['Ne'] = np.exp(df_states.iloc[:,env.sidx['logNe']]).sum(axis=1) #.sum(axis=0)
    df_stats['N_a'] = np.exp(df_states.iloc[:,[0,3]]).sum(axis=1) #.sum(axis=0)
    df_stats['N_i'] = np.exp(df_states.iloc[:,[1,4]]).sum(axis=1) #.sum(axis=0)
    df_stats['N_s'] = np.exp(df_states.iloc[:,[2,5]]).sum(axis=1) #.sum(axis=0)
    df_stats['t'] = df_states.iloc[:,-1]
    df_stats['q'] = np.exp(df_states.iloc[:,env.sidx['logq']])

# plot total N over time
import matplotlib
import matplotlib.pyplot as plt
# population size for each age class across time
x = np.arange(df_stats.shape[0])

plt.figure(figsize=(8, 5))
plt.plot(x, df_stats['totAngo'], label='Angostura', color='blue')
plt.plot(x, df_stats['totIsl'], label='Isleta', color='orange')
plt.plot(x, df_stats['totSA'], label='San Acacia', color='green')

plt.xlabel('Timestep Index')
plt.ylabel('Total Population Size by Reach')
plt.legend()
plt.title('Total Population Size by Reach Over Time')
plt.show()
