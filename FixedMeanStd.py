import pandas as pd
import numpy as np
class FixedMeanStd:
    "fixed mean and standard deviation for normalization for a given environment"
    def __init__(self, env):
        self.envID = env.envID
        if env.envID in ['Hatchery3.2.2','Hatchery3.2.4','Hatchery3.2.6','Hatchery3.3.3']:
            self.mean = np.array([10.34357939,  9.26967126, 10.12292189,  8.11184044,  7.52888496,
        6.34880387, 20.10945595, 10.78685875])
            self.var = np.array([2.22852155, 3.25747513, 3.32704338, 1.79404018, 2.40663774,
       2.78347831, 0.39975523, 1.44030081])**2
        if env.envID == 'Hatchery3.2.3':
            # load parameter samples from the file
            param_uncertainty_filename = 'uncertain_parameters_posterior_samples4POMDP.csv'
            param_uncertainty_df = pd.read_csv(param_uncertainty_filename)
            param_means = param_uncertainty_df.mean().values
            param_vars = param_uncertainty_df.var().values
            self.mean = np.array([10.34357939,  9.26967126, 10.12292189,  8.11184044,  7.52888496,
        6.34880387, 20.10945595, 10.78685875])
            self.mean = np.concatenate((self.mean, param_means))
            self.var = np.array([2.22852155, 3.25747513, 3.32704338, 1.79404018, 2.40663774,
       2.78347831, 0.39975523, 1.44030081])**2
            self.var = np.concatenate((self.var, param_vars))
        if env.envID == 'Hatchery3.2.5':
            self.mean = np.array([10.34357939,  9.26967126, 10.12292189,  8.11184044,  7.52888496,
        6.34880387, 12.20722173,20.10945595, 10.78685875])
            self.var = np.array([2.22852155, 3.25747513, 3.32704338, 1.79404018, 2.40663774,
       2.78347831, 0.22209442, 0.39975523, 1.44030081])**2
        if env.envID in ['Hatchery3.3.1','Hatchery3.3.2']:
            self.mean = np.array([10.34357939,  9.26967126, 10.12292189, 10.34357939,  9.26967126, 10.12292189,  8.11184044,  7.52888496,
        6.34880387, 12.20722173, 20.10945595, 10.78685875, 0.5])
            self.var = np.array([2.22852155, 3.25747513, 3.32704338, 2.22852155, 3.25747513, 3.32704338, 1.79404018, 2.40663774,
       2.78347831, 0.22209442, 0.39975523, 1.44030081,0.25])**2
        if env.envID in ['Hatchery3.3.2.2']:
            self.mean = np.array([10.34357939,  9.26967126, 10.12292189, 8.11184044,  7.52888496,
        6.34880387, 12.20722173, 20.10945595, 10.78685875, 0.5])
            self.var = np.array([2.22852155, 3.25747513, 3.32704338, 1.79404018, 2.40663774,
       2.78347831, 0.22209442, 0.39975523, 1.44030081,0.25])**2
        if env.envID == 'Hatchery3.4.1':
            self.mean = np.array([10.34357939,  9.26967126, 10.12292189, 10.34357939,  9.26967126, 10.12292189,  8.11184044,  7.52888496,
        6.34880387, 12.20722173, 12.20722173, 20.10945595, 10.78685875, 0.5])
            self.var = np.array([2.22852155, 3.25747513, 3.32704338, 2.22852155, 3.25747513, 3.32704338, 1.79404018, 2.40663774,
       2.78347831, 0.22209442, 0.22209442, 0.39975523, 1.44030081,0.25])**2

        self.stored_batch = []
        self.rolloutnum = 0
        self.updateN = 1000 # Number of samples to collect before updating the mean and variance

    def update(self):
        self.stored_batch = []
        self.rolloutnum = 0

    def normalize(self, x):
        # check if there is a variable self.envID
        if hasattr(self, 'envID'):
            if 'Hatchery3.3' in self.envID or 'Hatchery3.4' in self.envID:
                x[0:-1] = (x[0:-1] - self.mean[0:-1]) / np.sqrt(self.var[0:-1] + 1e-8)
                return x
            else:
                return (x - self.mean) / np.sqrt(self.var + 1e-8)
        else:
            return (x - self.mean) / np.sqrt(self.var + 1e-8)