import itertools
from scipy.stats import multivariate_hypergeom, norm, beta
import pickle
import numpy as np
from math import floor
import random
import pandas as pd
import sys
import os
from AR1 import AR1


class Hatchery3_2:
    """
    environment for the hatchery problem implementation-grade parameterization.
    Unlike Hatchery 3.1, uses effective population size as a genetic diversity variable.
    """
    def __init__(self,initstate,parameterization_set,discretization_set,LC_prediction_method):
        self.envID = 'Hatchery3.2'
        self.partial = True
        self.episodic = True
        self.absorbing_cut = True # has an absorbing state and the episode should be cut shortly after reaching it.
        self.discset = discretization_set
        self.contstate = False
        self.special_stacking = False

        # Define parameters
        # call in parameterization dataset csv
        # Read csv 'parameterization_env1.0.csv'
        # for reach index, 1 = angostura, 2 = isleta, 3 = san acacia
        self.parset = parameterization_set - 1
        parameterization_set_filename = 'parameterization_hatchery3.1.csv'
        paramdf = pd.read_csv(parameterization_set_filename)
        self.n_reach = 3
        self.alpha = paramdf['alpha'][self.parset] 
        self.alphavar = paramdf['alphavar'][self.parset]
        self.beta = paramdf['beta'][self.parset] 
        self.Lmean = paramdf['Lmean'][self.parset]
        self.mu = np.array([paramdf['mu_a'][self.parset],paramdf['mu_i'][self.parset],paramdf['mu_s'][self.parset]])
        self.sd = paramdf['sd'][self.parset]
        self.beta_2 = paramdf['beta_2'][self.parset]
        #self.beta_stk = paramdf['beta_stk'][self.parset]
        self.tau = paramdf['tau'][self.parset]
        self.r0 = paramdf['r0'][self.parset]
        self.r1 = paramdf['r1'][self.parset]
        self.delfall = np.zeros((2,self.n_reach)) # first row is alpha and second row is beta for beta distribution
        self.deldiff = np.zeros((2,self.n_reach)) # first row is alpha and second row is beta for beta distribution
        self.delfall[0] = np.array([paramdf['delfall_a'][self.parset], paramdf['delfall1_i'][self.parset], paramdf['delfall1_s'][self.parset]])
        self.delfall[1] = np.array([paramdf['delfall_a'][self.parset], paramdf['delfall2_i'][self.parset], paramdf['delfall2_s'][self.parset]])
        self.deldiff[0] = np.array([paramdf['deldiff_a'][self.parset], paramdf['deldiff1_i'][self.parset], paramdf['deldiff1_s'][self.parset]])
        self.deldiff[1] = np.array([paramdf['deldiff_a'][self.parset], paramdf['deldiff2_i'][self.parset], paramdf['deldiff2_s'][self.parset]])
        self.phifall = np.array([paramdf['phifall_a'][self.parset],paramdf['phifall_i'][self.parset],paramdf['phifall_s'][self.parset]])
        self.phidiff = np.array([paramdf['phidiff_a'][self.parset],paramdf['phidiff_i'][self.parset],paramdf['phidiff_s'][self.parset]])
        self.lM0mu = np.array([paramdf['lM0mu_a'][self.parset],paramdf['lM0mu_i'][self.parset],paramdf['lM0mu_s'][self.parset]])
        self.lM1mu = np.array([paramdf['lM1mu_a'][self.parset],paramdf['lM1mu_i'][self.parset],paramdf['lM1mu_s'][self.parset]])
        self.lMwmu = np.array([paramdf['lMwmu_a'][self.parset],paramdf['lMwmu_i'][self.parset],paramdf['lMwmu_s'][self.parset]])
        self.lM0sd = np.array([paramdf['lM0sd_a'][self.parset],paramdf['lM0sd_i'][self.parset],paramdf['lM0sd_s'][self.parset]])
        self.lM1sd = np.array([paramdf['lM1sd_a'][self.parset],paramdf['lM1sd_i'][self.parset],paramdf['lM1sd_s'][self.parset]])
        self.lMwsd = np.array([paramdf['lMwsd_a'][self.parset],paramdf['lMwsd_i'][self.parset],paramdf['lMwsd_s'][self.parset]])
        self.irphi = paramdf['irphi'][self.parset]
        self.p0 = paramdf['p0'][self.parset]
        self.p1 = paramdf['p1'][self.parset]
        self.sz = paramdf['sz'][self.parset]
        self.fpool_f = np.array([paramdf['fpoolf_a'][self.parset],paramdf['fpoolf_i'][self.parset],paramdf['fpoolf_s'][self.parset]])
        self.fpool_s = np.array([paramdf['fpools_a'][self.parset],paramdf['fpools_i'][self.parset],paramdf['fpools_s'][self.parset]])
        self.frun_f = np.array([paramdf['frunf_a'][self.parset],paramdf['frunf_i'][self.parset],paramdf['frunf_s'][self.parset]])
        self.frun_s = np.array([paramdf['fruns_a'][self.parset],paramdf['fruns_i'][self.parset],paramdf['fruns_s'][self.parset]])
        self.avgeff_fr = np.array([paramdf['avgeff_fr_a'][self.parset],paramdf['avgeff_fr_i'][self.parset],paramdf['avgeff_fr_s'][self.parset]])
        self.avgeff_fp = np.array([paramdf['avgeff_fp_a'][self.parset],paramdf['avgeff_fp_i'][self.parset],paramdf['avgeff_fp_s'][self.parset]])
        self.thetaf = np.array([paramdf['thetaf_a'][self.parset],paramdf['thetaf_i'][self.parset],paramdf['thetaf_s'][self.parset]])
        self.thetas = np.array([paramdf['thetas_a'][self.parset],paramdf['thetas_i'][self.parset],paramdf['thetas_s'][self.parset]])
        self.n_genotypes = paramdf['n_genotypes'][self.parset]
        self.n_locus = paramdf['n_locus'][self.parset]
        self.n_cohorts = paramdf['n_cohorts'][self.parset]
        self.b = np.array([paramdf['b1'][self.parset],paramdf['b2'][self.parset],paramdf['b3'][self.parset],paramdf['b4'][self.parset]])
        self.fc = np.array([paramdf['fc1'][self.parset],paramdf['fc2'][self.parset],paramdf['fc3'][self.parset],paramdf['fc4'][self.parset]])
        self.s0egg = paramdf['s0egg'][self.parset] # survival rate of egg to hatchery
        self.s0larvae = paramdf['s0larvae'][self.parset] # survival rate of hatchery to age 0 cohort
        self.eggcollection_max = paramdf['eggcollection_max'][self.parset] # maximum number of eggs that can be collected 
        self.larvaecollection_max = paramdf['larvaecollection_max'][self.parset] # maximum number of larvae that can be collected 
        self.sc = np.array([paramdf['s1'][self.parset],paramdf['s2'][self.parset],paramdf['s3'][self.parset]]) # cohort survival rate by age group
        self.Nth = paramdf['Nth'][self.parset]
        self.extant = paramdf['extant'][self.parset] # reward for not being
        self.prodcost = paramdf['prodcost'][self.parset] # production cost in spring if deciding to produce
        self.unitcost = paramdf['unitcost'][self.parset] # unit production cost.
        self.Ne2Nratio = paramdf['Ne2Nratio'][self.parset] # ratio of effective population size to total population size in hatchery
        # for monitoring simulation (alternate parameterization for )
        self.sampler = self.sz
        
        # discount factor
        self.gamma = 0.99

        # start springflow simulation model and springflow-to-"Larval carrying capacity" model.
        self.flowmodel = AR1()
        self.Otowi_minus_ABQ_springflow = self.flowmodel.constants[0] - self.flowmodel.constants[1] # difference between Otowi and ABQ springflow
        self.Otowi_minus_SA_springflow = self.flowmodel.constants[0] - self.flowmodel.constants[2] # difference between Otowi and San Acacia springflow
        self.LC_prediction_method = LC_prediction_method # 0=HMM, 1=GAM
        if self.LC_prediction_method == 0: # HMM
            self.LC_ABQ = pd.read_csv('springflow2LC_hmm_ABQ.csv')
            self.LC_SA = pd.read_csv('springflow2LC_hmm_San Acacia.csv')
        elif self.LC_prediction_method == 1: # linear GAM
            with open('LC_GAM_ABQ.pkl','rb') as handle:
                self.LC_ABQ = pickle.load(handle)
            with open('LC_GAM_San Acacia.pkl','rb') as handle:
                self.LC_SA = pickle.load(handle)

        # get some parameters for effective population size calculation
        ## get log(kappa) probability distribution and domain values
        with open('lkappa_dataset.pkl', 'rb') as handle:
            lkappa_dataset = pickle.load(handle)
        self.lkappa_prob = lkappa_dataset['lkappaprob']
        self.angolkappa_midvalues = lkappa_dataset['angolkappa_midvalues']
        self.isllkappa_midvalues = lkappa_dataset['isllkappa_midvalues']
        A,B             = np.meshgrid(self.angolkappa_midvalues,
                                    self.isllkappa_midvalues,
                                    indexing='ij')
        mask            = self.lkappa_prob > 1e-3            # keep only useful combos
        self.kappa_exp  = np.exp(np.stack([A[mask], B[mask], B[mask]], axis=-1))  # (nκ,3)
        self.kappa_prob = self.lkappa_prob[mask]             # (nκ,)

        ## calculate different alpha values (number of eggs produced per spawner) and their probabilities in the posterior distribution
        ## (used for calculating sigma^2_{b_w} in the document Hatchery 3.2).
        # define ends (95% CI) of the posterior distribution
        alphainterval = np.linspace(self.alpha - 1.96*np.sqrt(self.alphavar), self.alpha + 1.96*np.sqrt(self.alphavar), 11)
        # calculate the probability for each bin
        self.alphaprob = norm.cdf(alphainterval[1:], loc=self.alpha, scale=np.sqrt(self.alphavar)) - norm.cdf(alphainterval[:-1], loc=self.alpha, scale=np.sqrt(self.alphavar))
        self.alpha_centers = (alphainterval[:-1] + alphainterval[1:]) / 2
        self.alpha_vec  = np.hstack((self.alpha, self.alpha_centers))  # (nα,)
        ## get the probability distribution of river drying, delfall, and avg deldiff.
        bins = 5
        intervals = np.linspace(0,1,bins+1) # bin boundaries
        delfall_i = np.array([self.delfall[0,1],self.delfall[1,1]]) # isleta alpha beta
        delfall_s = np.array([self.delfall[0,2],self.delfall[1,2]]) # san acacia alpha beta
        delfallprob_i = beta.cdf(intervals[1:], a=delfall_i[0], b=delfall_i[1]) - beta.cdf(intervals[:-1], a=delfall_i[0], b=delfall_i[1])
        delfallprob_s = beta.cdf(intervals[1:], a=delfall_s[0], b=delfall_s[1]) - beta.cdf(intervals[:-1], a=delfall_s[0], b=delfall_s[1])
        delmid_values = (intervals[:-1] + intervals[1:]) / 2 # get the mid value for each bin
        deldiffavg = np.zeros(self.n_reach)
        deldiffavg[1:] = self.deldiff[0,1:] / np.sum(self.deldiff,axis=0)[1:] # average value for deldiff. angostura value is 0

        ## get expected survival rates for age1+ and age 0
        M0 = np.exp(self.lM0mu)
        M1 = np.exp(self.lM1mu)
        Mw = np.exp(self.lMwmu)
        combo = np.array(list(itertools.product(delmid_values, delmid_values)))
        self.combo_delfallprob = np.array(list(itertools.product(delfallprob_i, delfallprob_s)))
        self.combo_delfallprob = np.prod(self.combo_delfallprob, axis=1) # product of delfall probabilities
        valididx = np.where(self.combo_delfallprob > 1e-2)[0] # only keep the combinations with probability greater than 1e-3
        self.combo_delfallprob = self.combo_delfallprob[valididx] # only keep the valid combinations
        delfall = np.column_stack((np.zeros(combo.shape[0]),combo))
        delfall = delfall[valididx,:] # only keep the valid combinations
        self.sj = np.exp(-124*M0 - 150*Mw)*((1 - delfall) + self.tau*delfall*deldiffavg + (1 - self.tau)*self.r0*np.mean(self.phidiff))
        self.sa = np.exp(-124*M1 - 150*Mw)*((1 - delfall) + self.tau*delfall + (1 - self.tau)*self.r1*np.mean(self.phifall))
        self.sj =  np.prod(self.sj,axis=1)**(1/self.sj.shape[1]) # geometric mean of sj
        self.sa =  np.prod(self.sa,axis=1)**(1/self.sa.shape[1]) # geometric mean of sa

        ## get average of age 2+ fish 
        self.avgsa = np.sum(self.sa * self.combo_delfallprob)
        self.AVGage_of_age2plus = 2*(1/(1+self.avgsa)) + 3*(self.avgsa/(1+self.avgsa)) # assume that the fish only lives till age 3
        


        # range for each variables
        self.N0minmax = [0,1e7] 
        self.N1minmax = [0,1e7] # N1 and N1 minmax are the total population minmax.
        self.Nhminmax = [0, 300000]
        self.Nbminmax = [0,500]
        self.pminmax = [0, 50000] # Total number of fish stocked in a season that make it to breeding season
        self.qminmax = [self.flowmodel.flowmin[0], self.flowmodel.flowmax[0]] # springflow in Otowi (Otowi gauge)
        self.Neminmax = [0, 1e7]
        self.tminmax = [0,3]
        self.qhatminmax = [self.flowmodel.flowmin[0] + self.flowmodel.bias_95interval[0], self.flowmodel.flowmax[0] + self.flowmodel.bias_95interval[1]] # springflow estimate in otowi
        self.aminmax = [0, 300000]
        # dimension for each variables
        self.N0_dim = (self.n_reach)
        self.N1_dim = (self.n_reach)
        self.Nh_dim = (1)
        self.Nb_dim = (1)
        self.p_dim = (self.n_reach)
        self.q_dim = (1)
        self.Ne_dim = (1)
        self.t_dim = (1)
        self.qhat_dim = (1)
        self.statevar_dim = (self.N0_dim, self.N1_dim, self.Nh_dim, self.Nb_dim, self.p_dim, self.q_dim, self.Ne_dim, self.t_dim)
        self.obsvar_dim = (self.N0_dim, self.N1_dim, self.Nh_dim, self.Nb_dim, self.p_dim, self.qhat_dim, self.Ne_dim, self.t_dim)

        # starting 3.0, discretization for discrete variables and ranges for continuous variables will be defined in a separate function, state_discretization.
        discretization_obj = self.state_discretization(discretization_set)
        self.states = discretization_obj['states']
        self.observations = discretization_obj['observations']
        self.actions = discretization_obj['actions']

        # define how many discretizations each variable has.
        if self.discset == -1: 
            self.statespace_dim = list(np.ones(np.sum(self.statevar_dim))*-1) # continuous statespace is not defined (marked as -1)
            self.actionspace_dim = list(map(lambda x: len(x[1]), self.actions.items()))
            self.obsspace_dim = list(np.ones(np.sum(self.obsvar_dim))*-1)
        else:
            self.statespace_dim = np.concatenate((np.array(list(map(lambda x: len(x[1]), self.states.items())))*np.array([np.ones(self.statevar_dim[i]) for i in range(len(self.statevar_dim))],dtype='object')))
            #self.statespace_dim = list(np.array(
            #    list(map(lambda x: len(x[1]), self.states.items()))
            #) * np.array([
            #    self.N0_dim, self.N1_dim, self.Nh_dim, self.q_dim, self.p_dim, np.prod(self.ph_dim), self.ph0_dim, self.G_dim, self.t_dim
            #]))
            self.actionspace_dim = list(map(lambda x: len(x[1]), self.actions.items()))
            self.obsspace_dim = np.concatenate((np.array(list(map(lambda x: len(x[1]), self.observations.items())))*np.array([np.ones(self.obsvar_dim[i]) for i in range(len(self.obsvar_dim))],dtype='object')))
            #self.obsspace_dim = list(np.array(
            #    list(map(lambda x: len(x[1]), self.observations.items()))
            #) * np.array([
            #    self.N0_dim, self.N1_dim, self.Nh_dim, self.q_dim, 
            #    self.G_dim, self.t_dim
            #]))
            # for discrete springflow, get the index of the discrete springflow values in the LC to springflow mapping table.
            self.ABQq = np.minimum(np.maximum(self.states['q'] - self.Otowi_minus_ABQ_springflow, self.flowmodel.flowmin[1]), self.flowmodel.flowmax[1])
            self.SAq = np.minimum(np.maximum(self.states['q'] - self.Otowi_minus_SA_springflow, self.flowmodel.flowmin[2]), self.flowmodel.flowmax[2])
            if self.LC_prediction_method == 0:
                ABQqfrac = (self.ABQq - self.LC_ABQ['springflow'].values[0])/(self.LC_ABQ['springflow'].values[-1] - self.LC_ABQ['springflow'].values[0])
                SAqfrac = (self.SAq - self.LC_SA['springflow'].values[0])/(self.LC_SA['springflow'].values[-1] - self.LC_SA['springflow'].values[0])
                self.disc_sf_idxs_abq = np.round(ABQqfrac * (len(self.LC_ABQ['springflow']) - 1)).astype(int)
                self.disc_sf_idxs_sa = np.round(SAqfrac * (len(self.LC_SA['springflow']) - 1)).astype(int)

        # idx in the state and observation list for each state/observation variables
        self.sidx = {}
        self.oidx = {}
        self.aidx = {}
        for idx, key in enumerate(self.states.keys()):
            if idx ==0:
                self.sidx[key] = np.arange(0,self.statevar_dim[idx])
            else:
                self.sidx[key] = np.arange(np.sum(self.statevar_dim[0:idx]),np.sum(self.statevar_dim[0:idx]) + self.statevar_dim[idx])
        for idx, key in enumerate(self.observations.keys()):
            if idx ==0:
                self.oidx[key] = np.arange(0,self.obsvar_dim[idx])
            else:
                self.oidx[key] = np.arange(np.sum(self.obsvar_dim[0:idx]),np.sum(self.obsvar_dim[0:idx]) + self.obsvar_dim[idx])
        self.aidx = {key: idx for idx, key in enumerate(self.actions.keys())}

        # Initialize state and observation
        self.state, self.obs = self.reset(initstate)

    def reset(self, initstate=None):
        if type(initstate) is not np.ndarray:
            initstate = np.ones(len(self.statevar_dim))*-1
            initstate[-1] = 0 # start from spring if initstate is not provided
        
        # Initialize state variables
        new_state = []
        new_obs = []
        if initstate[-1] == -1:
            season  = np.random.choice([0,1]).astype(int) # only start from spring or fall angostura stocking.
        else:
            season = initstate[-1].astype(int)

        if season == 0: # spring
            # N0 & ON0
            N0val, N1val = self.init_pop_sampler()
            if initstate[0] == -1:
                # N0val = random.choices(list(np.arange(1, len(self.states['N0']))), k = self.statevar_dim[0])
                new_state.append(N0val) # don't start from the smallest population size
                new_obs.append(N0val)
            else:
                new_state.append(initstate[0])
                new_obs.append(initstate[0])
            # N1 & ON1
            if initstate[1] == -1:
                # N1val = random.choices(list(np.arange(1, len(self.states['N1']))), k = self.statevar_dim[1])
                new_state.append(N1val) # don't start from
                new_obs.append(N1val)
            else:
                new_state.append(initstate[1])
                new_obs.append(initstate[1])
            # Nh & ONh
            if initstate[2] == -1:
                Nhval = [0] # no fish in the hatchery in the beginning
                new_state.append(Nhval)
                new_obs.append(Nhval)
            else:
                new_state.append(initstate[2])
                new_obs.append(initstate[2])
            # Nb
            if initstate[3] == -1:
                Nbval = 0 # no production in the beginning
                if self.discset == -1:
                    Nbval = [np.log(Nbval + 1)]
                else:
                    Nbval = [self._discretize_idx(Nbval, self.states['Nb'])]
                new_state.append(Nbval)
                new_obs.append(Nbval)
            else:
                new_state.append(initstate[3])
                new_obs.append(initstate[3])
            # p
            if initstate[4] == -1:
                pval = np.ones(self.n_reach)*np.log(0 + 1) # nothing's been stocked in the beginning
                new_state.append(pval)
                new_obs.append(pval)
            else:
                new_state.append(initstate[4])
                new_obs.append(initstate[4])
            # q & qhat
            if initstate[5] == -1:
                if self.discset == -1:
                    qval = np.random.uniform(size=1)*(self.states['logq'][1] - self.states['logq'][0]) + self.states['logq'][0]
                    bias = np.clip(np.random.normal(loc=self.flowmodel.bias_mean, scale=self.flowmodel.bias_std, size=1),self.flowmodel.bias_95interval[0],self.flowmodel.bias_95interval[1])
                    qhatval = np.log(np.exp(qval) + bias)
                else:
                    qval = random.choices(list(np.arange(0,len(self.states['q']))), k = self.statevar_dim[3])
                    springflow = self.states['q'][qval[0]]
                    bias = np.clip(np.random.normal(loc=self.flowmodel.bias_mean, scale=self.flowmodel.bias_std, size=1),self.flowmodel.bias_95interval[0],self.flowmodel.bias_95interval[1])
                    qhatval = springflow + bias
                    qhatval = [self._discretize_idx(qhatval[0], self.observations['qhat'])]
                new_state.append(qval)
                new_obs.append(qhatval)
            else:
                new_state.append(initstate[5])
                new_obs.append(initstate[5])
        else: # fall
            # N0 & ON0
            N0val, N1val = self.init_pop_sampler()
            if initstate[0] == -1:
                #N0val = random.choices(list(np.arange(1, len(self.states['N0']))), k = self.statevar_dim[0])
                new_state.append(N0val) # don't start from the smallest population size
                new_obs.append(N0val)
            else:
                new_state.append(initstate[0])
                new_obs.append(initstate[0])
            # N1 & ON1
            if initstate[1] == -1:
                #N1val = random.choices(list(np.arange(1, len(self.states['N1']))), k = self.statevar_dim[1])
                new_state.append(N1val) # don't start from
                new_obs.append(N1val)
            else:
                new_state.append(initstate[1])
                new_obs.append(initstate[1])
            # Nh & ONh
            if initstate[2] == -1:
                if self.discset == - 1:
                    Nhval = np.log(np.random.uniform(size=1)*(self.Nhminmax[1] - self.Nhminmax[0]) + self.Nhminmax[0] + 1)
                else:
                    Nhval = random.choices(list(np.arange(0, len(self.states['Nh']))), k = self.statevar_dim[2])
                new_state.append(Nhval)
                new_obs.append(Nhval)
            else:
                new_state.append(initstate[2])
                new_obs.append(initstate[2])
            # Nb
            if initstate[3] == -1:
                Nbval = 0 # no production in the beginning
                if self.discset == -1:
                    Nbval = [np.log(Nbval + 1)]
                else:
                    Nbval = [self._discretize_idx(Nbval, self.states['Nb'])]
                new_state.append(Nbval)
                new_obs.append(Nbval)
            else:
                new_state.append(initstate[3])
                new_obs.append(initstate[3])
            # p
            if initstate[4] == -1:
                pval = np.ones(self.n_reach)*np.log(0 + 1) # nothing's been stocked in the beginning
                new_state.append(pval)
                new_obs.append(pval)
            else:
                new_state.append(initstate[4])
                new_obs.append(initstate[4])
            # q & qhat
            if initstate[5] == -1:
                # springflow not applicable in the fall season, but just pick a random springflow
                if self.discset == -1:
                    qval = np.random.uniform(size=1)*(self.states['logq'][1] - self.states['logq'][0]) + self.states['logq'][0]
                    bias = np.clip(np.random.normal(loc=self.flowmodel.bias_mean, scale=self.flowmodel.bias_std, size=1),self.flowmodel.bias_95interval[0],self.flowmodel.bias_95interval[1])
                    qhatval = np.log(np.exp(qval) + bias)
                else:
                    qval = random.choices(list(np.arange(0,len(self.states['q']))), k = self.statevar_dim[3])
                    springflow = self.states['q'][qval[0]]
                    bias = np.clip(np.random.normal(loc=self.flowmodel.bias_mean, scale=self.flowmodel.bias_std, size=1),self.flowmodel.bias_95interval[0],self.flowmodel.bias_95interval[1])
                    qhatval = springflow + bias
                    qhatval = [self._discretize_idx(qhatval[0], self.observations['qhat'])]
                new_state.append(qval)
                new_obs.append(qhatval)
            else:
                new_state.append(initstate[5])
                new_obs.append(initstate[5])
        # Ne & ONe
        if self.discset == -1:
            Neval = np.array([np.sum(np.exp(N0val)-1 + np.exp(N1val)-1)*0.6])
            #Neval,_,_ = self.NeCalc(np.exp(N0val)-1, np.exp(N1val)-1, pval[0], Nbval[0])
        else:
            Neval = np.array([np.sum(np.exp(N0val)-1 + np.exp(N1val)-1)*0.6])
            #Neval,_,_ = self.NeCalc(np.exp(N0val)-1, np.exp(N1val)-1, pval, Nbval)
            Neval = self._discretize_idx(Neval, self.states['Ne'])
        new_state.append(np.log(Neval+1))
        new_obs.append(np.log(Neval+1))
        # t & Ot
        new_state.append([season])
        new_obs.append([season])

        self.state = list(np.concatenate(new_state))
        self.obs = list(np.concatenate(new_obs))
        return self.state, self.obs

    def step(self, aidx):
        extra_info = {}
        if self.discset == -1:
            N0 = np.exp(np.array(self.state)[self.sidx['logN0']]) - 1
            N1 = np.exp(np.array(self.state)[self.sidx['logN1']]) - 1
            Nh = np.exp(np.array(self.state)[self.sidx['logNh']]) - 1
            Nb = np.exp(np.array(self.state)[self.sidx['logNb']]) - 1
            p = np.exp(np.array(self.state)[self.sidx['logp']]) - 1
            q = np.exp(np.array(self.state)[self.sidx["logq"]]) - 1
            Ne = np.exp(np.array(self.state)[self.sidx["logNe"]]) - 1
            t = np.array(self.state)[self.sidx['t']].astype(int)
            qhat = np.exp(np.array(self.obs)[self.oidx['logqhat']]) - 1
        else:
            N0 = np.array(self.states["N0"])[np.array(self.state)[self.sidx['N0']]]
            N1 = np.array(self.states["N1"])[np.array(self.state)[self.sidx['N1']]]
            Nh = np.array(self.states["Nh"])[np.array(self.state)[self.sidx['Nh']]]
            Nb = np.array(self.states["Nb"])[np.array(self.state)[self.sidx['Nb']]]
            p = np.array(self.states['p'])[np.array(self.state)[self.sidx['p']]]
            q = np.array(self.states["q"])[np.array(self.state)[self.sidx['q']]]
            Ne = np.array(self.states['Ne'])[np.array(self.state)[self.sidx['Ne']]]
            t = np.array(self.states['t'])[np.array(self.state)[self.sidx['t']]]
            qhat = np.array(self.observations['qhat'])[np.array(self.obs)[self.oidx['qhat']]]
        a = self.actions["a"][aidx]
        t_next = t + 1 if t < 3 else np.zeros(1).astype(int)
        totN0 = np.sum(N0)
        totN1 = np.sum(N1)
        totpop = totN0 + totN1
        if t == 1 and totpop/4000 < 0.5:
            foo = 0
        if totpop >= self.Nth:
            if t == 0: # sring
                # demographic stuff (reproductin and summer survival)
                delfall = np.concatenate(([self.delfall[0][0]],np.random.beta(self.delfall[0][1:],self.delfall[1][1:])))
                deldiff = np.concatenate(([self.deldiff[0][0]],np.random.beta(self.deldiff[0][1:],self.deldiff[1][1:])))
                L = self.q2LC(q)
                extra_info['L'] = L
                kappa = np.exp(self.beta*(L - self.Lmean) + np.random.normal(self.mu, self.sd))
                extra_info['kappa'] = kappa
                effspawner = N0 + self.beta_2*N1 # effective number of spawners
                P1 = (self.alpha*N0)/(1 + self.alpha*effspawner/kappa) # number of recruits produced by age 1 fish that newly became adults
                P2 = (self.alpha*self.beta_2*N1)/(1 + self.alpha*effspawner/kappa) # number of recruits produced by age 2+ fish
                P = (self.alpha*effspawner)/(1 + self.alpha*effspawner/kappa)
                M0 = np.exp(np.random.normal(self.lM0mu, self.lM0sd))
                M1 = np.exp(np.random.normal(self.lM1mu, self.lM1sd))
                genT = (np.sum(P1) + np.sum(P2)*self.AVGage_of_age2plus)/np.sum(P)  # generation time
                summer_mortality = np.exp(-124*M0)*((1 - delfall) + self.tau*delfall*deldiff + (1 - self.tau)*self.r0*self.phidiff)
                extra_info['summer_mortality'] = summer_mortality
                N0_next = np.minimum(P*np.exp(-124*M0)*((1 - delfall) + self.tau*delfall*deldiff + (1 - self.tau)*self.r0*self.phidiff),np.ones(self.n_reach)*self.N0minmax[1])
                N1_next = np.minimum((N0+N1)*np.exp(-215*M1)*((1-delfall) + self.tau*delfall + (1 - self.tau)*self.r1*self.phifall),np.ones(self.n_reach)*self.N1minmax[1])
                # hatchery stuff (Nh, Nc, Nc0)
                Nh_next = a + 10 # added 10 so that there's no small discrepancy that doesn't allow stocking the produced amount in the fall.
                Nb_next = 2*a/1000 # the value 1000 is Thomas' ballpark estimate of stock-ready fish produced per female  # 2*a/self.fc[1] # number of broodstock needed to produce 'a' fish self.fc[1]=number of fish produced per age 2 female broodstock, which is multiplied by 2 b/c you need male and female.
                p_next = np.zeros(self.n_reach)
                # hydrological stuff. No springflow in fall (stays the same)
                Ne_next, Neh, New = self.NeCalc0(N0,N1,p,Nb,genT,0)
                # reward & done
                reward = self.extant - self.prodcost if a > 0 else self.extant
                done = False
                # update state & obs
                if self.discset == -1:
                    logN0_next = np.log(N0_next+1)
                    logN1_next = np.log(N1_next+1)
                    logNh_next = [np.log(Nh_next+1)]
                    logNb_next = [np.log(Nb_next+1)]
                    logp_next = np.log(p_next+1)
                    logNe_next = np.log(Ne_next+1)
                    OlogNe_next = logNe_next
                    self.state = list(np.concatenate([logN0_next, logN1_next, logNh_next, logNb_next, logp_next, np.array(self.state)[self.sidx["logq"]], logNe_next, t_next]))
                    self.obs = list(np.concatenate([logN0_next, logN1_next, logNh_next, logNb_next, logp_next, np.array(self.obs)[self.oidx["logqhat"]], OlogNe_next, t_next]))
                else:
                    N0_next_idx = [self._discretize_idx(val, self.states['N0']) for val in N0_next]
                    N1_next_idx = [self._discretize_idx(val, self.states['N1']) for val in N1_next]
                    Nh_next_idx = [self._discretize_idx(Nh_next, self.states['Nh'])]
                    Nb_next_idx = [self._discretize_idx(Nb_next, self.states['Nb'])]
                    p_next_idx = [self._discretize_idx(p_next, self.states['p'])]
                    q_next_idx = np.array(self.state)[self.sidx['q']] # no change
                    Ne_next_idx = [self._discretize_idx(Ne_next, self.states['Ne'])]
                    ONe_next_idx = Ne_next_idx # observed Ne gets the same value as the state Ne
                    t_next_idx = t_next
                    qhat_next_idx = np.array(self.obs)[self.oidx['qhat']] # no change
                    self.state = list(np.concatenate([N0_next_idx, N1_next_idx, Nh_next_idx, Nb_next_idx, p_next_idx, q_next_idx, Ne_next_idx, t_next_idx]).astype(int))
                    self.obs = list(np.concatenate([N0_next_idx, N1_next_idx, Nh_next_idx, Nb_next_idx, p_next_idx, qhat_next_idx, ONe_next_idx, t_next_idx]).astype(int))
            elif t < 3: # fall stocking for angostura or isleta
                # demographic stuff
                if a - Nh <= 1e-7: # valid action choice. stocking is less than the hatchery population size.
                    N0_next = N0.copy()
                    N0_next[t-1] = np.minimum(N0_next[t-1] + a*self.irphi,self.N0minmax[1]) # stocking angostura (t=1) or isleta (t=2) in the fall
                    N1_next = N1.copy()
                    Nh_next = Nh[0] - a
                    p_next = p.copy()
                    p_next[t-1] = p[t-1] + a*self.irphi # stocking angostura (t=1) or isleta (t=2) in the fall
                    # reward & done
                    reward = self.extant
                    done = False
                else: # invalid action choice that results in heavy penalty with termination
                    N0_next = N1_next = np.zeros(self.n_reach)
                    Nh_next = 0
                    Nb_next = Nb.copy()
                    p_next = p.copy()
                    p_next[t-1] = p[t-1] + a*self.irphi # stocking angostura (t=1) or isleta (t=2) in the fall
                    # reward & done
                    reward = 0
                    done = True
                    print("Invalid action choice. Terminating the episode.")
                # hydrological stuff
                q_next = q # no springflow in fall (stays the same)
                qhat_next = qhat # no springflow in fall (stays the same)
                # update state & obs
                if self.discset == -1:
                    logN0_next = np.log(N0_next+1)
                    logN1_next = np.log(N1_next+1)
                    logNh_next = [np.log(Nh_next+1)]
                    logp_next = np.log(p_next+1)
                    self.state = list(np.concatenate([logN0_next, logN1_next, logNh_next, np.array(self.state)[self.sidx["logNb"]], logp_next, np.array(self.state)[self.sidx["logq"]], np.array(self.state)[self.sidx["logNe"]], t_next]))
                    self.obs = list(np.concatenate([logN0_next, logN1_next, logNh_next, np.array(self.obs)[self.oidx["OlogNb"]], logp_next,np.array(self.obs)[self.oidx["logqhat"]], np.array(self.obs)[self.oidx["OlogNe"]], t_next]))
                else:
                    N0_next_idx = [self._discretize_idx(val, self.states['N0']) for val in N0_next]
                    N1_next_idx = [self._discretize_idx(val, self.states['N1']) for val in N1_next]
                    Nh_next_idx = [self._discretize_idx(Nh_next, self.states['Nh'])]
                    Nb_next_idx = np.array(self.state)[self.sidx['Nb']] # no change
                    p_next_idx = [self._discretize_idx(Nh_next, self.states['p'])]
                    q_next_idx = np.array(self.state)[self.sidx['q']] # no change
                    Ne_next_idx = np.array(self.state)[self.sidx['Ne']] # no change
                    ONe_next_idx = np.array(self.obs)[self.oidx['ONe']] # no change
                    t_next_idx = t_next
                    qhat_next_idx = np.array(self.obs)[self.oidx['qhat']] # no change
                    self.state = list(np.concatenate([N0_next_idx, N1_next_idx, Nh_next_idx, Nb_next_idx, p_next_idx, q_next_idx, Ne_next_idx, t_next_idx]).astype(int))
                    self.obs = list(np.concatenate([N0_next_idx, N1_next_idx, Nh_next_idx, Nb_next_idx, p_next_idx, qhat_next_idx, ONe_next_idx, t_next_idx]).astype(int))
            else: # t=3
                if a - Nh <= 1e-7: # valid action choice. stocking is less than the hatchery population size.
                    Mw = np.exp(np.random.normal(self.lMwmu, self.lMwsd))
                    N0_next = N0.copy()
                    N0_next[t-1] = N0_next[t-1] + a*self.irphi # stocking san acacia (t=3) in the fall
                    N0_next = np.minimum(N0_next*np.exp(-150*Mw),np.ones(self.n_reach)*self.N0minmax[1]) # stocking san acacia (t=3) in the fall
                    N1_next = N1*np.exp(-150*Mw)
                    Nh_next = 0
                    p_next = p.copy()
                    p_next[t-1] = p[t-1] + a*self.irphi # stocking angostura (t=1) or isleta (t=2) in the fall
                    p_next = p_next*np.exp(-150*Mw)
                    Ne_next, Neh, New = self.NeCalc0(N0_next,N1_next,p_next,Nb,None,1)
                    extra_info['Ne'] = Ne_next
                    extra_info['Neh'] = Neh
                    extra_info['New'] = New

                    # reward & done
                    reward = self.extant
                    done = False
                else:
                    N0_next = N1_next = np.zeros(self.n_reach)
                    Nh_next = 0
                    p_next = p.copy()
                    p_next[t-1] = p[t-1] + a*self.irphi # stocking angostura (t=1) or isleta (t=2) in the fall
                    p_next = p_next*np.exp(-150*Mw)
                    # reward & done
                    reward = 0
                    done = True
                    print("Invalid action choice. Terminating the episode.")
                # hydrological stuff
                q_next, qhat_next = self.flowmodel.nextflowNforecast(q) # springflow and forecast in spring
                q_next = q_next[0][0]
                # update state & obs
                if self.discset == -1:
                    logN0_next = np.log(N0_next+1)
                    logN1_next = np.log(N1_next+1)
                    logNh_next = np.array([np.log(Nh_next+1)])
                    logp_next = np.log(p_next+1)
                    logq_next = np.array([np.log(q_next+1)])
                    logqhat_next = np.array([np.log(qhat_next+1)])
                    logNe_next = np.log(Ne_next+1)
                    OlogNe_next = logNe_next
                    self.state = list(np.concatenate([logN0_next, logN1_next, logNh_next, np.array(self.state)[self.sidx["logNb"]], logp_next, logq_next, logNe_next, t_next]))
                    self.obs = list(np.concatenate([logN0_next, logN1_next, logNh_next, np.array(self.obs)[self.oidx["OlogNb"]], logp_next, logqhat_next, OlogNe_next, t_next]))
                else:
                    N0_next_idx = [self._discretize_idx(val, self.states['N0']) for val in N0_next]
                    N1_next_idx = [self._discretize_idx(val, self.states['N1']) for val in N1_next]
                    Nh_next_idx = [self._discretize_idx(Nh_next, self.states['Nh'])]
                    p_next_idx = [self._discretize_idx(val, self.states['p']) for val in p_next]
                    q_next_idx = [self._discretize_idx(q_next, self.states['q'])]
                    qhat_next_idx = [self._discretize_idx(qhat_next, self.observations['qhat'])]
                    Ne_next_idx = [self._discretize_idx(Ne_next, self.states['Ne'])]
                    ONe_next_idx = Ne_next_idx # observed Ne gets the same value as the state Ne
                    t_next_idx = t_next
                    self.state = list(np.concatenate([N0_next_idx, N1_next_idx, Nh_next_idx, np.array(self.state)[self.sidx["Nb"]], p_next_idx, q_next_idx, Ne_next_idx, t_next_idx]).astype(int))
                    self.obs = list(np.concatenate([N0_next_idx, N1_next_idx, Nh_next_idx,  np.array(self.obs)[self.oidx["ONb"]], p_next_idx, qhat_next_idx, ONe_next_idx, t_next_idx]).astype(int))
        else: # extinct
            reward = 0
            done = True
        return reward, done, extra_info

    def state_discretization(self, discretization_set):
        """
        input: discretization id.
        output: dictionary with states, observations, and actions.
        """

        if discretization_set == 0:
            states = {
                "N0": list(np.linspace(self.N0minmax[0], self.N0minmax[1], 31)), # population size dim:(3)
                "N1": list(np.linspace(self.N1minmax[0], self.N1minmax[1], 31)), # population size (3)
                "Nh": list(np.linspace(self.Nhminmax[0], self.Nhminmax[1], 11)), # hatchery population size (1)
                "Nb": list(np.linspace(self.Nbminmax[0], self.Nbminmax[1], 11)), # hatchery population size (1)
                "p": list(np.linspace(self.pminmax[0], self.pminmax[1], 11)), # hatchery population size (3)
                "q": list(np.linspace(self.qminmax[0], self.qminmax[1], 11)), # spring flow in Otowi (1)
                "Ne": list(np.linspace(self.Neminmax[0], self.Neminmax[1], 11)), # heterozygosity (1)
                "t": list(np.arange(self.tminmax[0], self.tminmax[1] + 1, 1)), # time (1)
            }

            observations = {
                "ON0": states['N0'],
                "ON1": states['N1'], 
                "ONh": states['Nh'],
                "ONb": states['Nb'],
                "Op": states['p'],
                "qhat": list(np.linspace(self.qhatminmax[0], self.qhatminmax[1] + 1, 11)), # forecasted flow in Otowi dim:(1)
                "ONe": states['Ne'],
                "Ot": states['t'], 
            }
            actions = {
                "a": list(np.linspace(self.aminmax[0], self.aminmax[1], 11)), # stocking/production size (1)
            }
        elif discretization_set == -1: # continuous
            states = {
                "logN0": list(np.log(np.array(self.N0minmax)+1)), # log population size dim:(3)
                "logN1": list(np.log(np.array(self.N1minmax)+1)), # log population size (3)
                "logNh": list(np.log(np.array(self.Nhminmax)+1)), # log hatchery population size (1)
                "logNb": list(np.log(np.array(self.Nbminmax)+1)), # log number of broodstock used for production
                "logp": list(np.log(np.array(self.pminmax)+1)), # Total number of fish stocked in a season that make it to breeding season (3)
                "logq": list(np.log(np.array(self.qminmax)+1)), # log spring flow in Otowi (Otowi) (1)
                "logNe": self.Neminmax, # heterozygosity (1)
                "t": self.tminmax # time (1)
            }
            observations = {
                "OlogN0": states['logN0'],
                "OlogN1": states['logN1'],
                "OlogNh": states['logNh'],
                "OlogNb": states['logNb'],
                "Ologp": states['logp'],
                "logqhat": list(np.log(np.array(self.qhatminmax)+1)), # forecasted flow in Otowi dim:(1)
                "OlogNe": states['logNe'],
                "Ot": states['t'],
            }
            actions = {
                "a": list(np.linspace(self.aminmax[0], self.aminmax[1], 31)), # stocking/production size (1)
            }
        return {'states': states,'observations': observations,'actions': actions}

    def init_pop_sampler(self):
        """
        output:
            (if continuous) population size: list of length n_reach (using dirichlet distribution)
            OR
            (if discrete) index of population size: list of index of population size
        """
        if self.discset == -1:
            init_totpop = np.exp(np.random.uniform(size=1)[0]*(self.states['logN0'][1] - np.log(self.Nth+1)) + np.log(self.Nth+1)) # initial total population size (don't let it be lowest state)
            init_ageprop = np.random.uniform(size=1)[0] # initial age proportion
            init_prop0 = init_ageprop * np.random.dirichlet(np.ones(self.n_reach),size=1)[0] # initial proportion for age 0
            init_prop1 = (1 - init_ageprop) * np.random.dirichlet(np.ones(self.n_reach),size=1)[0] # initial proportion for age 1+
            init_pop0 = init_prop0*init_totpop
            init_pop1 = init_prop1*init_totpop
            if any(init_pop0 + init_pop1 < self.Nth):
                init_pop0 = np.maximum(init_pop0, (self.Nth + 100) - init_pop1)
            return np.log(init_pop0+1), np.log(init_pop1+1)
        else:
            init_totpop_idx = random.choices(np.arange(1,len(self.states['N0'])),k=1)[0] # initial total population size (don't let it be lowest state)
            init_ageprop = np.random.uniform(size=1)[0] # initial age proportion
            init_prop0 = init_ageprop * np.random.dirichlet(np.ones(self.n_reach),size=1)[0] # initial proportion for age 0
            init_prop1 = (1 - init_ageprop) * np.random.dirichlet(np.ones(self.n_reach),size=1)[0] # initial proportion for age 1+
            init_pop0 = init_prop0*self.states['N0'][init_totpop_idx]
            init_pop1 = init_prop1*self.states['N0'][init_totpop_idx]
            return self.pop_discretize(init_pop0,init_pop1,init_totpop_idx)
        
    def pop_discretize(self, pop0, pop1, totpop_idx):
        """
        intput:
            popX: list of population size of age 0 or 1+ for the 3 reaches.
            totpop: total population size
        output:
            index of population size for age 0 and age 1+
        Uses Knuth's “rounding to a given sum” trick and runs in O(nlogn) due to one sort.
        """
        freq0 = pop0/self.states['N0'][totpop_idx]
        freq1 = pop1/self.states['N0'][totpop_idx]
        scaledfreq0 = freq0*((totpop_idx+1) - 1)
        scaledfreq1 = freq1*((totpop_idx+1) - 1)
        scaledfreq0_flr = np.floor(scaledfreq0)
        scaledfreq1_flr = np.floor(scaledfreq1)
        margin = np.sum(scaledfreq0_flr) + np.sum(scaledfreq1_flr) - np.sum(scaledfreq0) - np.sum(scaledfreq1)
        if margin < 0:
            scaledfrac0 = scaledfreq0 - scaledfreq0_flr
            scaledfrac1 = scaledfreq1 - scaledfreq1_flr
            scaledfrac = np.concatenate((scaledfrac0,scaledfrac1))
            new_scaledfreq_flr = np.concatenate((scaledfreq0_flr,scaledfreq1_flr))
            new_scaledfreq_flr[scaledfrac.argsort()[::-1][0:np.abs(round(margin))]] += 1
            scaledfreq0_flr = list(new_scaledfreq_flr[0:len(scaledfreq0_flr)].astype(int))
            scaledfreq1_flr = list(new_scaledfreq_flr[len(scaledfreq0_flr):(len(scaledfreq0_flr)+len(scaledfreq1_flr))].astype(int))
        return scaledfreq0_flr, scaledfreq1_flr

    def _discretize(self, x, possible_states):
        # get the 2 closest values in the possible_states to x 
        # and then get the weights disproportionate to the distance to the x
        # then use those weights as probabilities to chose one of the two states.
        if x < possible_states[0]:
            return possible_states[0]
        elif x > possible_states[-1]:
            return possible_states[-1]
        else:
            lower, upper = np.array(possible_states)[np.argsort(abs(np.array(possible_states) - x))[:2]]
            weights = np.array([upper - x, x - lower])/(upper - lower)  
            return random.choices([lower, upper], weights=weights,k=1)[0]
        
    def _discretize_idx(self, x, possible_states):
        # get the 2 closest values in the possible_states to x
        # and then get the weights disproportionate to the distance to the x
        # then use those weights as probabilities to chose one of the two states.
        if x <= possible_states[0]:
            return 0
        elif x >= possible_states[-1]:
            return len(possible_states) - 1
        else:
            frac = (x - np.array(possible_states)[0])/(np.array(possible_states)[-1] - np.array(possible_states)[0])
            fracscaled = frac*(len(possible_states) - 1)
            lower_idx = np.floor(fracscaled).astype(int)
            upper_idx = np.ceil(fracscaled).astype(int)
            if lower_idx == upper_idx:
                return lower_idx
            else:
                weights = np.array([upper_idx - fracscaled, fracscaled - lower_idx])
            return random.choices([lower_idx, upper_idx], weights=weights,k=1)[0]


    def q2LC(self, q):
        """
        input:
            q: springflow in Otowi (1)
            method: 0=HMM, 1=GAM
        output:
            LC: larval carrying capacity (3)
        """
        if self.LC_prediction_method == 0: # hmm
            if self.discset == -1: # continuous
                # get springflow at ABQ and SA for given springflow at Otowi
                abqsf = np.minimum(np.maximum(q - self.Otowi_minus_ABQ_springflow, self.flowmodel.flowmin[1]), self.flowmodel.flowmax[1])
                sasf = np.minimum(np.maximum(q - self.Otowi_minus_SA_springflow, self.flowmodel.flowmin[2]), self.flowmodel.flowmax[2])
                # get the index of the springflow in the LC to springflow mapping table
                abqsf_idx = np.round((abqsf - self.LC_ABQ['springflow'][0])/(self.LC_ABQ['springflow'].iloc[-1] - self.LC_ABQ['springflow'][0]) * (len(self.LC_ABQ['springflow']) - 1)).astype(int)
                sasf_idx = np.round((sasf - self.LC_SA['springflow'][0])/(self.LC_SA['springflow'].iloc[-1] - self.LC_SA['springflow'][0]) * (len(self.LC_SA['springflow']) - 1)).astype(int)
                # get the potential LC values for the given springflow
                LCsamplesABQ = self.LC_ABQ.iloc[abqsf_idx]
                LCsamplesSA = self.LC_SA.iloc[sasf_idx]
            else: # discrete
                # use pre-computed index of springflow in the LC to springflow mapping table
                LCsamplesABQ = self.LC_ABQ.iloc[self.disc_sf_idxs_abq[np.array(self.state)[self.sidx['q']]]]
                LCsamplesSA = self.LC_SA.iloc[self.disc_sf_idxs_sa[np.array(self.state)[self.sidx['q']]]]
            angoisletaLC = np.random.choice(LCsamplesABQ.values[0], size=1)
            saLC = np.random.choice(LCsamplesSA.values[0], size=1)
            L = np.array([angoisletaLC, saLC, saLC]).T[0]
        elif self.LC_prediction_method == 1: # gam
            # calculate the error for the LC prediction
            angoisletaLC_error = np.clip(np.random.normal(0, self.LC_ABQ['std']), -1.96*self.LC_ABQ['std'], 1.96*self.LC_ABQ['std'])
            saLC_error = np.clip(np.random.normal(0, self.LC_SA['std']), -1.96*self.LC_SA['std'], 1.96*self.LC_SA['std'])
            if self.discset == -1:
                # get springflow at ABQ and SA for given springflow at Otowi
                abqsf = np.minimum(np.maximum(q - self.Otowi_minus_ABQ_springflow, self.flowmodel.flowmin[1]), self.flowmodel.flowmax[1])
                sasf = np.minimum(np.maximum(q - self.Otowi_minus_SA_springflow, self.flowmodel.flowmin[2]), self.flowmodel.flowmax[2])
                # predict the LC using the GAM model
                angoisletaLC = np.maximum(self.LC_ABQ['model'].predict(abqsf) + angoisletaLC_error, 0) # make sure LC is not negative
                saLC = np.maximum(self.LC_SA['model'].predict(sasf) + saLC_error,0) # make sure LC is not negative
            else:
                angoisletaLC = np.maximum(self.LC_ABQ['model'].predict(self.ABQq[np.array(self.state)[self.sidx['q']]]) + angoisletaLC_error, 0) # make sure LC is not negative
                saLC = np.maximum(self.LC_SA['model'].predict(self.SAq[np.array(self.state)[self.sidx['q']]]) + saLC_error, 0) # make sure LC is not negative
            L = np.array([angoisletaLC, saLC, saLC]).T[0]
        return L
    
    def monitoring_sample(self):
        """
        simulate fall monitoring catch data from the model state
        output:
            monitoring catch data for each site (5, 6, 9 sites for angostura, isleta, and san acacia respectively)
        the values for number of sites is from 'Spring augmnetation planning.pdf'
        the average effort value is from pop_model_param_output.R, variable avgeff_f
        """

        sitenum = np.array([5, 6, 9])
        #sitenum = np.array([1000, 1000, 1000])
        avg_effort = [1257.0977, 891.8601, 1850.5951] # average area sampled at each site (square meters)
        if self.discset == -1:
            aidx = [self.sidx['logN0'][0],self.sidx['logN1'][0]]
            iidx = [self.sidx['logN0'][1],self.sidx['logN1'][1]]
            sidx = [self.sidx['logN0'][2],self.sidx['logN1'][2]]
            popsize = np.array([np.sum(np.exp(np.array(self.state)[aidx])), np.sum(np.exp(np.array(self.state)[iidx])), np.sum(np.exp(np.array(self.state)[sidx]))])
        else:
            aidx = [self.sidx['N0'][0],self.sidx['N1'][0]]
            iidx = [self.sidx['N0'][1],self.sidx['N1'][1]]
            sidx = [self.sidx['N0'][2],self.sidx['N1'][2]]
            popsize = np.array([np.sum(np.array(self.states['N0'])[np.array(self.state)[[0,3]]]), np.sum(np.array(self.states['N0'])[np.array(self.state)[[1,4]]]), np.sum(np.array(self.states['N0'])[np.array(self.state)[[2,5]]])])
        # calculate cpue from popsize
        avgp = np.mean([self.p0, self.p1])
        avgfallf = (self.fpool_f + self.frun_f)/2 # is the proportion of RGSM in the river segment exposed to sampling
        avgcatch = popsize*avgp*avgfallf*self.thetaf
        p = self.sampler / (avgcatch + self.sampler)
        cpue = np.array([np.random.negative_binomial(self.sampler, p[i], size=sitenum[i])/(self.avgeff_fr[i]+self.avgeff_fp[i])*100 for i in range(len(sitenum))], dtype=object) # cpue is catch per 100square meters.

        # test code below (calculates mean cpue for each reach for different reach population sizes)
        #sitenum = np.array([1000, 1000, 1000])
        #popsizes = [10e2, 10e3, 10e4, 10e5, 10e6, 10e7]
        #mcpue = np.zeros((len(popsizes), self.n_reach))
        #for j in range(len(popsizes)):
        #    popsize = np.ones(3) * popsizes[j]
        #    avgp = np.mean([self.p0, self.p1])
        #    avgfallf = (self.fpool_f + self.frun_f)/2 # is the proportion of RGSM in the river segment exposed to sampling
        #    avgcatch = popsize*avgp*avgfallf*self.thetaf
        #    p = self.sampler / (avgcatch + self.sampler)
        #    cpue = np.array([np.random.negative_binomial(self.sampler, p[i], size=sitenum[i])/avg_effort[i]*100 for i in range(len(sitenum))], dtype=object) # cpue is catch per 100square meters.
        #    mcpue[j] = np.array([np.mean(cpue[i]) for i in range(len(sitenum))], dtype=object) # mean cpue for each reach
        #print(self.fpool_f)
        #print(self.frun_f)
        return cpue#, mcpue
    
    def production_target(self):
        """
        quantifies the amount of fish to produce in the hatchery based on the forecast
        the values here for the model is from 'Spring augmnetation planning.pdf'
        """
        qhat = np.exp(np.array(self.obs)[self.oidx['logqhat']]) if self.discset == -1 else np.array([self.observations['qhat'][self.obs[self.oidx['qhat'][0]]]])
        qhat_kaf = qhat[0]/1233480.0 # convert cubic meter to kaf
        X = np.array([1,qhat_kaf])
        V = np.array([[1.662419546,-3.284657e-03],[-0.003284657,7.883848e-06]])
        se = np.sqrt(X@V@X.T)
        fit = -0.005417*(qhat_kaf) + 2.321860
        production_target = 1/(1 + np.exp(-(fit + 1.739607*se)))*299000 # the glm model predicts the percentage of the max population capacity which was 299000 in the planning document
        aidx = self._discretize_maxstock_idx(production_target, self.actions['a'],1)
        return aidx
    
    def stocking_decision(self):
        """
        quantifies how many fish to stock in each reach based on the monitoring data
        """
        mdata = self.monitoring_sample() # monitoring catch per effort data.
        stock = np.zeros(self.n_reach)
        augment = 0
        reachlen = np.array([12333473,8748359,8527714])/100 # this is a length in 100m^2 because cpue is in 100m^2
        # figure out which reach needs augmentation and how much
        for i in range(self.n_reach):
            meanCPUE = np.mean(mdata[i])
            #if len(np.where(mdata[i] > 0)[0]) > np.floor(len(mdata[i]/2)): # within a reach, are >= 50% of the sites occupied?
            if meanCPUE > 1.0: # is the reach-wide average CPUE >= 1.0?
                augment = 0 # no augmentation needed
            else:
                augment = 1
            #else:
            #    augment = 1
            if augment == 1:
                stock[i] = (1 - meanCPUE) * reachlen[i]
        
        nh = np.exp(self.state[self.sidx['logNh'][0]]) if self.discset == -1 else self.states['Nh'][self.state[self.sidx['Nh'][0]]] # amount of fish in the hatchery
        # discretize the stocking amount to the nearest action choices
        max_aidx = self._discretize_maxstock_idx(nh, self.actions['a'],0) # action index of the maximum amonut of fish that can be stocked in one reach, provided that the rest gets none.
        if np.sum(stock) > self.actions['a'][max_aidx]: # if the amount needed is greater than the hatchery population size, then stock in the proportion of the need for each reach
            stock = stock/np.sum(stock) * self.actions['a'][max_aidx]
        if max_aidx == 0:
            return list(np.zeros(self.n_reach).astype(int))
        else:
            return self._discretize_stocking_idx(max_aidx, stock)

    def _discretize_maxstock_idx(self, x, possible_actions, lower_or_uppper):
        '''
        same as _discretize_idx but for actions, when x is in between possible_actions, it always returns the lower one.
        if lower_or_upper = 0, then it returns the lower one.
        if lower_or_upper = 1, then it returns the upper one.
        '''
        if x <= possible_actions[0]:
            return 0
        elif x >= possible_actions[-1]:
            return len(possible_actions) - 1
        else:  
            frac = (x - np.array(possible_actions)[0])/(np.array(possible_actions)[-1] - np.array(possible_actions)[0])
            fracscaled = frac*(len(possible_actions) - 1)
            if lower_or_uppper == 1:
                return np.ceil(fracscaled).astype(int)
            else:
                return np.floor(fracscaled).astype(int)
    
    def _discretize_stocking_idx(self, max_aidx, stock):
        # discretize the stocking amount to the nearest action choices
        freq = stock/self.actions['a'][max_aidx]
        stock_scaled = freq*max_aidx
        stock_scaled_flr = np.floor(stock_scaled)
        margin = np.sum(stock_scaled_flr) - np.sum(stock_scaled)
        if margin < 0:
            scaledfrac = stock_scaled - stock_scaled_flr
            stock_scaled_flr[np.argsort(scaledfrac)[::-1][0:np.abs(round(margin))]] += 1
        return list(stock_scaled_flr.astype(int))
    
    def _compute_mask(self,states=None):
        """Return 1/0 vector of length N_ACTIONS indicating legal choices."""
        if states is None:
            states = np.array(self.obs).reshape(1,-1)
        springidx = states[:,self.oidx['Ot'][0]] == 0
        fallidx = ~springidx
        mask = np.zeros((states.shape[0], len(self.actions['a'])), dtype=int)
        # in spring, you can't produce more than what the broodstock size can produce
        if np.sum(springidx) > 0:
            mask[springidx] = np.ones(len(self.actions['a'])).astype(int)
        if np.sum(fallidx) > 0: # in fall, you cannot stock more than what's in the hatchery.
            hatchery = np.atleast_2d(np.exp(states[np.ix_(np.flatnonzero(fallidx),self.oidx['OlogNh'])]))
            falllegal = hatchery - 1 >= self.actions['a']
            mask[fallidx] = falllegal
        return mask.astype(int)

    def NeCalc0(self, N0, N1, p, Nb, genT, season):
        """
        Calculate the effective population size (Ne). 
        intput:
            p: total number of fish stocked and survived to breeding season
            Nb: number of broodstock used for production
            N0: total population size of age 0 fish (1)
            N1: total population size of age 1+ fish (1)
            kappa: larval carrying capacity (3)
            season: 0= spring, 1=fall, in spring, only Ne = New, in fall Ne = f(New,Neh)
        output: Ne value 
        """
        if season == 0: # spring
            # calculate wild population's Ne
            totN0, totN1  = N0.sum(), N1.sum()
            effspawner    = N0 + self.beta_2*N1                      # (3,)
            alph             = self.alpha_vec[:,None,None]              # (nα,1,1)
            kappa             = self.kappa_exp.T[None,:,:]               # (1,3,nκ)
            denom         = 1 + alph*effspawner[:,None]/kappa               # (nα,3,nκ)
            f1            = (alph*N0[:,None]/denom).sum(1)/totN0        # (nα,nκ)
            fa            = (alph*self.beta_2*N1[:,None]/denom).sum(1)/totN1
            bvals         = (self.sj[:,None,None]*( (f1*totN0+fa*totN1)/(totN0+totN1) )).transpose(1,0,2) # (nα,nκ,ndel)
            b             = bvals[0]                                 # mean-α row
            recruitvar    = ((bvals[1:] - b)**2 * self.alphaprob[:,None,None]).sum(0)
            grate         = self.sa[:,None] + b/2
            var_dg        = self.sa[:,None]*(1-self.sa[:,None]) + b/4 + recruitvar/4
            factor = (var_dg/(grate**2) * self.kappa_prob * self.combo_delfallprob[:,None]).sum()
            New = (totN0+totN1)/factor
            New = np.array([New / genT]) # generation time adjusted wild Ne.
            return New, 0, New
        else:
            New = np.exp(self.state[self.sidx['logNe'][0]]) - 1
            # calculate hatchery population's Ne
            if np.sum(p) == 0: # if no fish are stocked, then Ne = New
                Ne = np.array([New])
                Neh = np.array([0])
            else:
                stocked_cont = np.sum(p)
                total_cont = np.sum(N0)
                x = stocked_cont/(total_cont + stocked_cont)
                #effspawner = N0 + self.beta_2*N1 # effective number of spawners
                #stocked_cont = np.sum((self.alpha*p)/(1 + self.alpha*effspawner/kappa)) # stocked fish contribution
                #total_cont =  np.sum((self.alpha*(effspawner))/(1 + self.alpha*effspawner/kappa)) # wild fish contribution
                #x = stocked_cont/(total_cont)
                #mu_k = self.fc[1]*self.irphi*np.exp(-150*np.prod(np.exp(self.lMwmu))**(1/3))
                #Neh = np.maximum(mu_k*(2*Nb - 1)/4, 0) # variance effective population size of hatchery population
                Neh = Nb * self.Ne2Nratio
                # apply Ryman-Laikre effect to calculate effective population size
                Ne = 1/(x**2/Neh + (1-x)**2/(New))
            return Ne, Neh, New