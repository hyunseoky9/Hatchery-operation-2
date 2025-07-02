from call_paramset import call_paramset, call_env
import numpy as np
import torch
import torch.multiprocessing as mp
from choose_action import choose_action
from choose_action_a3c import choose_action_a3c
def calc_performance(env, device, rms, fstack, policy, episodenum=1000, t_maxstep = 1000):
    """
    same as calc_performance.py but built for ddpg and td3 algorithms. 
    non-parallelized version.
    calculate the performance of the agent in the environment.
    For DQN calculate performance with the Q network. (use Q variable in the function input)
    For Tabular Q learning and value iteration, calculate perofrmance using the policy table.
    For policy gradient methods, calculate performance using the policy network.
    """
    print('serial calc_performance called')
    if env.envID in ['Hatchery3.2.2']:
        actiondist = np.zeros(len(env.actionspace_dim)) # distribution of actions taken
        actiondistcount = 0
    avgrewards = 0

    for i in range(episodenum):
        rewards = 0
        env.reset()
        stack = rms.normalize(env.obs)*fstack if rms is not None else env.obs*fstack
        done = False
        t = 0
        while done == False:
            state = rms.normalize(env.obs) if rms is not None else env.obs
            stack = np.concatenate((stack[len(env.obs):], env.obs))

            with torch.no_grad():                                   # <– no grads here
                s = torch.as_tensor(state.copy(), dtype=torch.float32, device=device).unsqueeze(0)
                action = policy(s)
                action = action.cpu().numpy().squeeze(0)
                if env.envID in ['Hatchery3.2.2']:
                    actiondist += action
                    actiondistcount += 1
            _, reward, done, _ = env.step(action)
            rewards += reward
            if t >= (t_maxstep - 1):
                done = True
            t += 1
        avgrewards += rewards
    if env.envID in ['Hatchery3.2.2']:
        actiondist = actiondist/actiondistcount
        print(np.round(actiondist,2))
    return avgrewards/episodenum