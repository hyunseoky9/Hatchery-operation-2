import numpy as np
def stacking(env, stack, next_state):
    """
    This function takes a stack and a next state as input and returns the updated stack.
    For some environments, a state gets stacked only under some, so this function is used to manage that.

    **This function is same as the one in stacking.py but it assumes the stack is a numpy array instead of a list. (used for ddpg)
    """
    return np.concatenate((stack[len(next_state):], next_state))