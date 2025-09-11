from scipy.special import logit, expit
import pickle
import numpy as np
import random
class whitenoise_normalized:
    """
    White noise (ARMA(0,0)) model for simulating springflow after learning that it does better than AR(1) on transformed springflow
    models angostura and san acacia springflow independently.
    """
    def __init__(self):

        # load parameters
        with open('white_noise_params_ABQ.pkl', 'rb') as handle:
            abqdf = pickle.load(handle)
        with open('white_noise_params_San Acacia.pkl', 'rb') as handle:
            sanacaciadf = pickle.load(handle)

        self.flowmin = np.array([abqdf['flowmin'], sanacaciadf['flowmin']])
        self.flowmax = np.array([abqdf['flowmax'], sanacaciadf['flowmax']])
        self.allowedmin = self.flowmin * 0.9
        self.allowedmax = self.flowmax * 1.1
        self.constants = np.array([abqdf['constant'], sanacaciadf['constant']])
        self.abqparams = {
            'mu': abqdf['mu'],
            'std': abqdf['std'],
            'flowmin': abqdf['flowmin'],
            'flowmax': abqdf['flowmax'],
            'allowed_flowmin': abqdf['flowmin']*0.9,
            'allowed_flowmax': abqdf['flowmax']*1.1
        }
        self.saparams = {
            'mu': sanacaciadf['mu'],
            'std': sanacaciadf['std'],
            'flowmin': sanacaciadf['flowmin'],
            'flowmax': sanacaciadf['flowmax'],
            'allowed_flowmin': sanacaciadf['flowmin']*0.9,
            'allowed_flowmax': sanacaciadf['flowmax']*1.1
        }

    def nextflow(self,q):
        '''
        Generate the next time step of flow for both gages.
        * it's actually wrong to model two gages separately. but the RL environment (Hatchery3.x) only takes the first argument
        '''
        abq_initial = np.random.normal(self.abqparams['mu'], self.abqparams['std'])
        sa_initial = np.random.normal(self.saparams['mu'], self.saparams['std'])
        # back transform
        abq_flow = expit(abq_initial) * (self.abqparams['allowed_flowmax'] - self.abqparams['allowed_flowmin']) + self.abqparams['allowed_flowmin']
        sa_flow = expit(sa_initial) * (self.saparams['allowed_flowmax'] - self.saparams['allowed_flowmin']) + self.saparams['allowed_flowmin']
        return np.array([abq_flow, sa_flow])