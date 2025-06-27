def stacking(env, stack, next_state):
    """
    This function takes a stack and a next state as input and returns the updated stack.
    For some environments, a state gets stacked only under some, so this function is used to manage that.
    """
    try:
        if env.special_stacking == True:
            if env.envID in ['Hatchery3.0','Hatchery3.1','Hatchery3.2']:
                if env.state[env.sidx['t'][0]] in [0,1]:
                    return stack[len(next_state):] + next_state
                else:
                    return stack
    except:
        return stack[len(next_state):] + next_state