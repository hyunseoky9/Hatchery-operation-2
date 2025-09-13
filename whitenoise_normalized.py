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
        with open('white_noise_params_Otowi.pkl', 'rb') as handle:
            otowidf = pickle.load(handle)
        self.flowmin = np.array([abqdf['flowmin'], sanacaciadf['flowmin']])
        self.flowmax = np.array([abqdf['flowmax'], sanacaciadf['flowmax']])
        self.allowedmin = self.flowmin * 0.9
        self.allowedmax = self.flowmax * 1.1
        self.otowiconstant = otowidf['constant']
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
        self.otowiparams = {
            'mu': otowidf['mu'],
            'std': otowidf['std'],
            'flowmin': otowidf['flowmin'],
            'flowmax': otowidf['flowmax'],
            'allowed_flowmin': otowidf['flowmin']*0.9,
            'allowed_flowmax': otowidf['flowmax']*1.1
        }
        # load forecast bias parameters
        with open('nrcs_forecast_bias_stats.pkl', 'rb') as handle:
            self.bias_params = pickle.load(handle)
        self.bias_mean = self.bias_params['mean_bias']
        self.bias_std = self.bias_params['std_bias']
        self.bias_95interval = [self.bias_mean - 1.96 * self.bias_std, self.bias_mean + 1.96 * self.bias_std]



    def nextflow(self,q):
        '''
        Generate the next time step of flow for both gages.
        * it's actually wrong to model two gages separately. but the RL environment (Hatchery3.x) only takes the first argument
        '''
        abq_initial = np.random.normal(self.abqparams['mu'], self.abqparams['std'])
        # back transform
        abq_flow = expit(abq_initial) * (self.abqparams['allowed_flowmax'] - self.abqparams['allowed_flowmin']) + self.abqparams['allowed_flowmin']
        return np.array([abq_flow, -1]) # -1 is just a placeholder
        
    def nextflowNforecast(self):
        '''Generate the next time step of flow for both gages and apply forecast bias.
        '''
        abq_initial = np.random.normal(self.abqparams['mu'], self.abqparams['std'])
        # back transform
        abq_flow = expit(abq_initial) * (self.abqparams['allowed_flowmax'] - self.abqparams['allowed_flowmin']) + self.abqparams['allowed_flowmin']

        # otowi forecast with bias
        bias = np.clip(
            np.random.normal(loc=self.bias_mean, scale=self.bias_std, size=1),
            self.bias_95interval[0],
            self.bias_95interval[1]
        )

        # get otowi flow from abq flow using constant difference
        otowi_flow = abq_flow - (self.constants[0] - self.otowiconstant)
        # apply forecast bias
        forecast = otowi_flow + bias
        # make sure forecast is within range
        forecast = np.maximum(forecast, self.otowiparams['flowmin'] - self.bias_95interval[1])
        forecast = np.minimum(forecast, self.otowiparams['flowmax'] + self.bias_95interval[1])
        forecast = np.max(forecast, 0)
    
        return np.array([abq_flow, forecast])
