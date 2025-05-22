
import pickle
import numpy as np
import random
class AR1:
    '''
    model that simulates spring flow
    (total volume from march thru july) at 3 gauges 
    (otowi, abq, san acacia) using AR1 model fitted
    to the 1997-2024 gauge data.
    '''
    def __init__(self):
        # gauge order
        self.order = {'otowi': 0, 'abq': 1, 'sanacacia': 2}

        # load parameters
        with open('ar1_params.pkl', 'rb') as handle:
            self.params = pickle.load(handle)
        self.constants = np.array([self.params['otowiconst'], self.params['abqconst'], self.params['sanacaciaconst']])
        self.phi_1 = self.params['phi_1']
        self.sigma2 = self.params['sigma2']
        self.flowmin = self.params['flowmin']
        self.flowmax = self.params['flowmax']
        
        # load bias parameters
        with open('nrcs_forecast_bias_stats.pkl', 'rb') as handle:
            self.bias_params = pickle.load(handle)
        self.bias_mean = self.bias_params['mean_bias']
        self.bias_std = self.bias_params['std_bias']
        self.bias_95interval = [self.bias_mean - 1.96 * self.bias_std, self.bias_mean + 1.96 * self.bias_std]

    def nextflow(self, otowispringflow):
        # input: otowi springflow last year
        # output: this year springflow at 3 otowi, abq, san acacia
        # we will prevent the prediction of Otowi springflow from going below the minimum or above the maximum springflow observed in the past.
        # For other gauges, the difference between the Otowi and the other gauges is constant, and the predicted springflow at these gauges when Otowi's
        # springflow is min or max roughly falls close to the min and max of these gauge's springflow.
        Otowival = np.maximum(self.constants[0] + (self.phi_1 * otowispringflow + np.random.normal(0, np.sqrt(self.sigma2))), self.flowmin[0])
        Otowival = np.minimum(Otowival, self.flowmax[0])
        ABQval = Otowival + self.constants[1] - self.constants[0]
        SAval = Otowival + self.constants[2] - self.constants[0]
        vals = np.array([Otowival, ABQval, SAval])
        return vals
    
    def nextflowNforecast(self, otowispringflow):
        # input: otowi springflow last year
        # output: this year springflow at 3 otowi, abq, san acacia + nrcs springflow forecast at otowi
        # we will prevent the prediction from going below the minimum springflow observed in the past for each gauge.
        Otowival = np.maximum(self.constants[0] + (self.phi_1 * otowispringflow + np.random.normal(0, np.sqrt(self.sigma2))), self.flowmin[0])
        Otowival = np.minimum(Otowival, self.flowmax[0])
        ABQval = Otowival + self.constants[1] - self.constants[0]
        SAval = Otowival + self.constants[2] - self.constants[0]
        vals = np.array([Otowival, ABQval, SAval])

        bias = np.clip(
            np.random.normal(loc=self.bias_mean, scale=self.bias_std, size=1),
            self.bias_95interval[0],
            self.bias_95interval[1]
        )
        
        forecast = vals[self.order['otowi']] + bias
        forecast = np.maximum(forecast, self.flowmin - self.bias_95interval[1])
        forecast = np.minimum(forecast, self.flowmax + self.bias_95interval[1])

        # make sure forecast is not negative
        forecast = np.max(forecast, 0)
        return vals, forecast
