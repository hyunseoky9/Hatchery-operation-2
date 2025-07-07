import numpy as np
def relative_logpopsize(env):
    """
    Calculate the relative log population size in a given state of the env.
    """
    logpopmax = np.log(1e7)
    logpopsize = np.log(np.exp(env.state[env.sidx['logN0']]) + np.exp(env.state[env.sidx['logN1']]) + 1)
    logpop = logpopsize / np.log(env.N0minmax[1])
    return logpop
