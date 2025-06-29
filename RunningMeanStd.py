import numpy as np
class RunningMeanStd:
    def __init__(self, shape, updateN, momentum=0.999, eps=1e-4):
        self.mean  = np.zeros(shape, np.float32)
        self.var   = np.ones(shape,  np.float32)
        self.count = eps
        self.m     = momentum
        self.stored_batch = []
        self.rolloutnum = 0
        self.updateN = updateN # Number of samples to collect before updating the mean and variance

    def update(self):
        # Chen et al. parallel batch normalization
        x = np.array(self.stored_batch)
        batch_mean = x.mean(0)
        batch_var  = x.var(0)
        batch_count = x.shape[0]

        delta      = batch_mean - self.mean
        tot_count  = self.count + batch_count

        new_mean   = self.mean + delta * batch_count / tot_count
        m_a        = self.var * self.count
        m_b        = batch_var * batch_count
        M2         = m_a + m_b + delta**2 * self.count * batch_count / tot_count
        new_var    = M2 / tot_count

        self.mean, self.var, self.count = \
            new_mean * self.m + self.mean * (1-self.m), \
            new_var  * self.m + self.var  * (1-self.m), \
            tot_count

        # empty the stored batch
        self.stored_batch = []
        self.rolloutnum = 0

    def normalize(self, x):
        return (x - self.mean) / np.sqrt(self.var + 1e-8)