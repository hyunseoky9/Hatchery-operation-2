import numpy as np
class FixedMeanStd:
    "fixed mean and standard deviation for normalization for a given environment"
    def __init__(self, env):
        if env.envID == 'Hatchery3.2.2':
            self.mean = np.array([10.34357939,  9.26967126, 10.12292189,  8.11184044,  7.52888496,
        6.34880387, 20.10945595, 10.78685875])
            self.var = np.array([2.22852155, 3.25747513, 3.32704338, 1.79404018, 2.40663774,
       2.78347831, 0.39975523, 1.44030081])**2
        self.stored_batch = []
        self.rolloutnum = 0
        self.updateN = 1000 # Number of samples to collect before updating the mean and variance

    def update(self):
        self.stored_batch = []
        self.rolloutnum = 0

    def normalize(self, x):
        return list((x - self.mean) / np.sqrt(self.var + 1e-8))