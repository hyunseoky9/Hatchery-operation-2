import argparse
import time
import torch
import os 
import numpy as np
from Hatchery3_3_7 import Hatchery3_3_7
import copy
import torch
from stacking2 import *
import os
import pickle

def load_episodes(runid):
    # read perfgapcalc_task.txt
    with open('perfgapcalc_task.txt', 'r') as f:
        lines = f.readlines()
    # load files under the runid 
    runid = str(runid)
    base_dir = './human_play_results/'
    pickle_filenames = []
    in_target_run = False


    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        if line.startswith('runid='):
            in_target_run = (line.split('=', 1)[1].strip() == runid)
            continue
        if in_target_run:
            pkl_path = line if os.path.isabs(line) else os.path.join(base_dir, line)
            pickle_filenames.append(pkl_path)

    episodes = []
    for pkl_file in pickle_filenames:
        with open(pkl_file, 'rb') as f:
            data = pickle.load(f)
        if isinstance(data, list):
            episodes.extend(data)
        else:
            episodes.append(data)

    return pickle_filenames, episodes
    

def calc_performance_gap(runid):
    # setting parameters
    ## performance gap calculation parameters
    horizon = 20 # 10 years
    iterations = 20#1000
    ## RL policy parameters
    config = {'seed': 983543, 'paramset': 163}

    # get list of episodes to work on
    pickle_filenames, episodes = load_episodes(runid)
    
    # load the policy
    envobsvars = ['logcpue_r_sep','numsamples_r_sep','logcatch_r_sep']
    cval = 1
    env = Hatchery3_3_7(None,1,-1,1,1,{'c':cval,'no_genetics':0,'obsvars': envobsvars,'sample_all_months':True,'sample_multiplier':2}) # initstate,parameterizationID,discretization,LCpredmethod
    wd = f'./TD3 results/good_ones/seed{config["seed"]}_paramset{config["paramset"]}'
    policy_filename = f"{wd}/bestPolicyNetwork_{env.envID}_par{env.parset}_dis{env.discset}_TD3.pt"
    rmsfilename = f"{wd}/rms_{env.envID}_par{env.parset}_dis{env.discset}_TD3.pkl"
    if os.path.exists(rmsfilename):
        standardize = True
        with open(rmsfilename, "rb") as f:
            rms = pickle.load(f)
    device = torch.device('cpu')  # Force CPU usage
    Policy = torch.load(policy_filename, weights_only=False)
    Policy = Policy.to(device) 
    fstack = 1 if not hasattr(Policy, 'fstack') else Policy.fstack
    print(f'number of episodes loaded: {len(episodes)}')
    for epi, ep in enumerate(episodes):
        # calculate L1 and euclidean distance between human and RL actions.
        ep['L1_a_dist'] = []
        ep['euclidean_a_dist'] = []
        simsteps = np.arange(len(ep['envcheckpoints']))
        ep['performance_gap'] = []
        print(f'calculating performance gap for {len(simsteps)} steps')
        for simstep in simsteps:
            if (ep['states'][simstep][-1] == 0) or (simstep == simsteps[-1]): # spring step, skip
                ep['performance_gap'].append((None, None, None)) # add None for spring steps
                ep['L1_a_dist'].append(None) # set action distance to None for spring steps
                ep['euclidean_a_dist'].append(None) # set action distance to None for spring steps
                continue
            ep['L1_a_dist'].append(np.abs(ep['actions'][simstep] - ep['RLactions'][simstep]).sum())
            ep['euclidean_a_dist'].append(np.sqrt(((ep['actions'][simstep] - ep['RLactions'][simstep])**2).sum()))
            # calculate performance gap
            env_og = ep['envcheckpoints'][simstep]
            V_human = 0
            V_pi = 0
            for i in range(2): # i=0 is human and i=1 is RL for first action.
                for _ in range(iterations):
                    envinstance = copy.deepcopy(env_og)
                    newstate = rms.normalize(envinstance.obs.copy()) if standardize else envinstance.obs.copy()
                    stack = np.concatenate([newstate] * fstack)
                    done = False
                    t = 0
                    rewards = 0
                    while done == False:
                        with torch.no_grad():
                            state = torch.tensor(stack, dtype=torch.float32, device=device).unsqueeze(0)  # Add batch dimension
                            action = Policy(state).cpu().numpy().flatten()
                        if t == 0:
                            if i == 0:
                                action = ep['actions'][simstep]
                            else:
                                action = ep['RLactions'][simstep]
                        _, reward, done, info = envinstance.step(action)
                        newstate  = rms.normalize(envinstance.obs.copy()) if standardize else envinstance.obs.copy()
                        stack = stacking(envinstance,stack,newstate)
                        rewards += reward*(envinstance.gamma**t)
                        t += 1
                        if t >= horizon:
                            done = True
                if i == 0:
                    V_human = rewards
                else:
                    V_pi = rewards
            delV = V_human - V_pi
            ep['performance_gap'].append((delV, V_human, V_pi))
        # save the updated episodes with performance gap and action distance metrics
        ofilename = pickle_filenames[epi].replace(".pkl", "_perfgap_updated.pkl") # add perfgap_updated to filename
        with open(ofilename, "wb") as f:
            pickle.dump(episodes, f)                    
                
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Calculate performance gap for a given runid')
    parser.add_argument('--runid', type=str, required=True, help='Run ID to process')
    args = parser.parse_args()
    
    t0 = time.time()
    calc_performance_gap(args.runid)
    #calc_performance_gap(1) # for testing, use runid=1
    elapsed = time.time() - t0
    print(f"Done. Elapsed time: {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")