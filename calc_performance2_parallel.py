from call_paramset import call_paramset, call_env
import random
import numpy as np
import torch
import torch.multiprocessing as mp
from choose_action import choose_action
from choose_action_a3c import choose_action_a3c
def calc_performance_parallel(env, device, seed, config, rms, fstack, policy=None, episodenum=1000, t_maxstep=1000):
    """
    same as calc_performance-parallel.py but built for ddpg and td3 algorithms. 
    parallelized version
    calculate the performance of the agent in the environment.
    For DQN calculate performance with the Q network. (use Q variable in the function input)
    For Tabular Q learning and value iteration, calculate perofrmance using the policy table.
    For policy gradient methods, calculate performance using the policy network.
    """
    print('parallel calc_performance called')
    action_size = len(env.actionspace_dim)
    # for getting policy related info
    policy_info = {
    'actiondist': torch.zeros(action_size).share_memory_(),  # Shared memory array
    'actiondistcount': mp.Value('i', 0),
    'lock': mp.Lock()  # Lock to ensure thread safety
    }
    # set environment configuration for the workers
    initlist = None
    #config = "{'init': %s,'paramset': %d, 'discretization': %d}" %(str(initlist),env.parset+1,env.discset)
    envinit_params = {'envconfig': config, 'envid': env.envID}

    # worker parameters
    #################
    num_workers = 4
    #################
    if policy is not None:
        policy.share_memory()
    total_rewards = mp.Value('f', 0.0)


    # Set multiprocessing method
    mp.set_start_method("spawn", force=True)
    # start timer
    processes = []
    # divide the number of epsiodes by workers.
    episodenum_perworker = [int(np.floor(episodenum/num_workers))]*(num_workers - 1)
    episodenum_perworker = episodenum_perworker + [int(episodenum - np.sum(episodenum_perworker))]
    # Spawn worker processes
    for worker_id in range(num_workers):
        p = mp.Process(target=worker, args=(policy, episodenum_perworker[worker_id], rms, worker_id, envinit_params, t_maxstep, fstack, action_size, device, policy_info, total_rewards,seed))
        p.start()
        processes.append(p)
    
    # Wait for all processes to finish
    for p in processes:
        p.join()

    if env.envID in ['Hatchery3.2.2', 'Hatchery3.2.3']:
        print('Action Distribution:')
        print(np.round((policy_info['actiondist']/policy_info['actiondistcount'].value).numpy(),2))

    return total_rewards.value/episodenum




def worker(policy, workerepisodenum, rms, worker_id, envinit_params, t_maxstep, fstack, action_size, device, policy_info, total_rewards,seed):
    # set seed
    worker_seed = seed + (worker_id + 1)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    torch.manual_seed(worker_seed)

    # set env
    env = call_env(envinit_params)

    totrewards_perworker = 0 # set reward counter.

    if env.envID in ['Hatchery3.2.2', 'Hatchery3.2.3']:
        actiondist = torch.zeros(action_size, dtype=torch.float32)
        actiondistcount = 0

    for i in range(workerepisodenum):
        rewards = 0
        env.reset()
        stack = np.concatenate([rms.normalize(env.obs)]*fstack) if rms is not None else np.concatenate([env.obs]*fstack)
        done = False
        t = 0
        while done == False:
            state = rms.normalize(env.obs) if rms is not None else env.obs
            stack = np.concatenate((stack[len(env.obs):], env.obs))

            with torch.no_grad():                                   # <– no grads here
                s = torch.as_tensor(state.copy(), dtype=torch.float32, device=device).unsqueeze(0)
                action = policy(s)
                action = action.cpu().numpy().squeeze(0)
                if env.envID in ['Hatchery3.2.2', 'Hatchery3.2.3']:
                    actiondist += action
                    actiondistcount += 1
            _, reward, done, _ = env.step(action)
            rewards += reward
            if t >= (t_maxstep - 1):
                done = True
            t += 1
        totrewards_perworker += rewards

    # additional policy related info update
    with policy_info['lock']:  
        policy_info['actiondist'].add_(actiondist) # Directly update shared tensor
        policy_info['actiondistcount'].value += actiondistcount
        
    # upload total rewards    
    with total_rewards.get_lock(): 
        total_rewards.value += totrewards_perworker # update global counter

    