import numpy as np
from absorbing import *
from stacking import *

def pretrain(env, nq, memory, max_steps, batch_size, PrioritizedReplay, max_priority, postterm_len, fstack, standardize, rms):
    # Make a bunch of random actions from a random state and store the experiences
    reset = True
    memadd = 0 # number of transitions added to memory
    n = nq.n
    stepcounter = 0
    while memadd < batch_size:
        if reset == True:
            if env.partial == False:
                env.reset()
                state = rms.normalize(env.state) if standardize else env.state
            else:
                env.reset()
                state = rms.normalize(env.obs) if standardize else env.obs
            previous_action = 0
            reset = False
            stack = state*fstack
            t = 0
            termination_t = 0
        # Make a random action
        action = np.random.choice(np.flatnonzero(env._compute_mask()))
        true_state = env.state
        reward, done, _ = env.step(action)

        if env.episodic == False and env.absorbing_cut == True: # if continuous task and absorbing state is defined
            if absorbing(env,true_state) == True: # terminate shortly after the absorbing state is reached.
                termination_t += 1
                if termination_t >= postterm_len: # run x steps once in absorbing state and then terminate
                    done = True
        if t >= max_steps:
            done = True
        t += 1
        stepcounter += 1
        if env.partial == False:
            if standardize:
                rms.stored_batch.append(env.obs)
                normstate = rms.normalize(env.obs)
                rms.rolloutnum += 1
                if rms.rolloutnum >= 10:
                    rms.update()
            else:
                normstate = env.obs
            next_state = normstate
        else:
            if standardize:
                rms.stored_batch.append(env.obs)
                normstate = rms.normalize(env.obs)
                rms.rolloutnum += 1
                if rms.rolloutnum >= 10:
                    rms.update()  # update the running mean and variance after collecting enough samples
            else:
                normstate = env.obs
            next_state = normstate
            
        next_stack = stacking(env,stack,next_state)

        if done:
            reset = True
            if stepcounter >= 12:
                nq.add(stack, action, reward, next_stack, done, previous_action, memory, PrioritizedReplay)
                memadd += n
        else:
            if env.episodic == False and memadd == (batch_size - 1): # if continuous task AND this is the last transition to be added on the memory AND it's not done, make it done for marking episode ends properly.
                done = True
            if stepcounter >= 12:
                # increase memadd by 1 if nq is full
                if len(nq.queue) == n-1:
                    memadd += 1
                nq.add(stack, action, reward, next_stack, done, previous_action, memory, PrioritizedReplay)
            state = next_state
            stack = next_stack
            previous_action = action
    nq.queue = [] # clear the n-step queue
    nq.rqueue = [] 
