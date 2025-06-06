import pickle
import numpy as np
from math import floor
import random
import pandas as pd
import sys
import os
from AR1 import AR1


class Hatchery3_1_test:
    """
    Hatchery3.1 adapted to test how heterozygosity is affected by the augmentation if we  breed from only 1 pair
    """
    def __init__(self,initstate,parameterization_set,discretization_set,LC_prediction_method):
        self.envID = 'Hatchery3.1'
        self.partial = True
        self.episodic = True
        self.absorbing_cut = True # has an absorbing state and the episode should be cut shortly after reaching it.
        self.discset = discretization_set
        self.contstate = False
        self.special_stacking = True

        # Define parameters
        # call in parameterization dataset csv
        # Read csv 'parameterization_env1.0.csv'
        # for reach index, 1 = angostura, 2 = isleta, 3 = san acacia
        self.parset = parameterization_set - 1
        parameterization_set_filename = 'parameterization_hatchery3.1.csv'
        paramdf = pd.read_csv(parameterization_set_filename)
        self.n_reach = 3
        self.alpha = paramdf['alpha'][self.parset] 
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
        self.thetaf = np.array([paramdf['thetaf_a'][self.parset],paramdf['thetaf_i'][self.parset],paramdf['thetaf_s'][self.parset]])
        self.thetas = np.array([paramdf['thetas_a'][self.parset],paramdf['thetas_i'][self.parset],paramdf['thetas_s'][self.parset]])
        self.n_genotypes = paramdf['n_genotypes'][self.parset]
        self.n_locus = paramdf['n_locus'][self.parset]
        self.n_cohorts = paramdf['n_cohorts'][self.parset]
        self.b = np.array([0,1,0,0])
        #self.b = np.array([paramdf['b1'][self.parset],paramdf['b2'][self.parset],paramdf['b3'][self.parset],paramdf['b4'][self.parset]])
        self.fc = np.ones(self.n_cohorts)*1000000
        #self.fc = np.array([paramdf['fc1'][self.parset],paramdf['fc2'][self.parset],paramdf['fc3'][self.parset],paramdf['fc4'][self.parset]])
        self.s0egg = paramdf['s0egg'][self.parset] # survival rate of egg to hatchery
        self.s0larvae = paramdf['s0larvae'][self.parset] # survival rate of hatchery to age 0 cohort
        self.eggcollection_max = paramdf['eggcollection_max'][self.parset] # maximum number of eggs that can be collected 
        self.larvaecollection_max = paramdf['larvaecollection_max'][self.parset] # maximum number of larvae that can be collected 
        self.sc = np.array([paramdf['s1'][self.parset],paramdf['s2'][self.parset],paramdf['s3'][self.parset]]) # cohort survival rate by age group
        self.Nth = paramdf['Nth'][self.parset]
        self.extant = paramdf['extant'][self.parset] # reward for not being
        self.prodcost = paramdf['prodcost'][self.parset] # production cost in spring if deciding to produce
        self.unitcost = paramdf['unitcost'][self.parset] # unit production cost.
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
        # range for each variables
        self.N0minmax = [0,10e7] 
        self.N1minmax = [0,10e7] # N1 and N1 minmax are the total population minmax.
        self.Nhminmax = [0, 300000]
        self.Ncminmax = [0, (self.eggcollection_max+self.larvaecollection_max)*4] # Nc is the number of larvae in the hatchery.
        self.Nc0minmax = [0, (self.eggcollection_max+self.larvaecollection_max)*4]
        self.qminmax = [self.flowmodel.flowmin[0], self.flowmodel.flowmax[0]] # springflow in Otowi (Otowi gauge)
        self.pminmax = [0,1]
        self.phminmax = [0,1]
        self.ph0minmax = [0,1]
        self.pcminmax = [0,1]
        self.Gminmax = [0, 1]
        self.tminmax = [0,3]
        self.qhatminmax = [self.flowmodel.flowmin[0] + self.flowmodel.bias_95interval[0], self.flowmodel.flowmax[0] + self.flowmodel.bias_95interval[1]] # springflow estimate in otowi
        self.aminmax = [0, 300000]
        # dimension for each variables
        self.N0_dim = (self.n_reach)
        self.N1_dim = (self.n_reach)
        self.Nh_dim = (1)
        self.Nc_dim = (self.n_cohorts)
        self.Nc0_dim = (1)
        self.q_dim = (1)
        self.p_dim = (self.n_locus, self.n_genotypes)
        self.ph_dim = (self.n_cohorts, self.n_locus, self.n_genotypes)
        self.ph0_dim = (self.n_locus, self.n_genotypes)
        self.pc_dim = (self.n_locus, self.n_genotypes)
        self.G_dim = (1)
        self.t_dim = (1)
        self.qhat_dim = (1)
        self.statevar_dim = (self.N0_dim, self.N1_dim, self.Nh_dim, self.Nc_dim, self.Nc0_dim, self.q_dim, np.prod(self.p_dim), np.prod(self.ph_dim), np.prod(self.ph0_dim), np.prod(self.pc_dim), self.G_dim, self.t_dim)
        self.obsvar_dim = (self.N0_dim, self.N1_dim, self.Nh_dim, self.Nc_dim, self.Nc0_dim, self.qhat_dim, self.G_dim, self.t_dim)

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
                Nhval = [0] # no fish in the hatchery in the beginning of the spring.
                new_state.append(Nhval)
                new_obs.append(Nhval)
            else:
                new_state.append(initstate[2])
                new_obs.append(initstate[2])
            # Nc & ONc
            if initstate[3] == -1:
                Ncval = 2 #np.random.uniform(size=4)*self.eggcollection_max*self.s0egg + np.random.uniform(size=4)*self.larvaecollection_max*self.s0larvae
                Ncval = np.ceil(np.cumprod(np.concatenate([[1],self.sc])) * Ncval)
                if self.discset == -1:
                    Ncval = np.log(Ncval + 1)
                else:
                    Ncval = [self._discretize_idx(val, self.states['Nc']) for val in Ncval]
                new_state.append(Ncval)
                new_obs.append(Ncval)
            else:
                new_state.append(initstate[3])
                new_obs.append(initstate[3])
            # Nc0 & ONc0
            if initstate[4] == -1:
                Nc0val = [0] 
                new_state.append(Nc0val)
                new_obs.append(Nc0val)
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
                new_state.append(initstate[3])
                new_obs.append(initstate[3])
            # p
            if initstate[6] == -1:
                pval = self.init_genotype_freq_sampler()
                new_state.append(pval)
            else:
                new_state.append(initstate[4])
            # ph
            if initstate[7] == -1:
                phval = np.concatenate([self.init_genotype_freq_sampler() for _ in range(self.n_cohorts)])
                new_state.append(phval)
            else:
                new_state.append(initstate[5])
            # ph0
            if initstate[8] == -1:
                ph0val = list(np.zeros(self.n_genotypes*self.n_locus).astype(int))
                new_state.append(ph0val)
            else:
                new_state.append(initstate[6])
            # pc
            if initstate[9] == -1:
                pcval = list(np.zeros(self.n_genotypes*self.n_locus).astype(int))
                new_state.append(pcval)
            else:
                new_state.append(initstate[7])
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
            # Nc & ONc
            if initstate[3] == -1:
                Ncval = 2 #np.random.uniform(size=4)*self.eggcollection_max*self.s0egg + np.random.uniform(size=4)*self.larvaecollection_max*self.s0larvae
                Ncval = np.ceil(np.cumprod(np.concatenate([[1],self.sc])) * Ncval)
                if self.discset == -1:
                    Ncval = np.log(Ncval + 1)
                else:
                    Ncval = [self._discretize_idx(val, self.states['Nc']) for val in Ncval]
                new_state.append(Ncval)
                new_obs.append(Ncval)
            else:
                new_state.append(initstate[3])
                new_obs.append(initstate[3])
            # Nc0 & ONc0
            if initstate[4] == -1:
                Nc0val = np.random.uniform(size=1)*self.eggcollection_max*self.s0egg + np.random.uniform(size=1)*self.larvaecollection_max*self.s0larvae
                if self.discset == -1:
                    Nc0val = np.log(Nc0val + 1)
                else:
                    Nc0val = [self._discretize_idx(Nc0val[0], self.states['Nc0'])]
                new_state.append(Nc0val)
                new_obs.append(Nc0val)
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
                new_state.append(initstate[3])
                new_obs.append(initstate[3])
            # p
            if initstate[6] == -1:
                pval = self.init_genotype_freq_sampler()
                new_state.append(pval)
            else:
                new_state.append(initstate[4])
            # ph
            if initstate[7] == -1:
                phval = np.concatenate([self.init_genotype_freq_sampler() for _ in range(self.n_cohorts)]) # assume that genotype frequency is the same in all cohorts to the wild population initially.
                new_state.append(phval)
            else:
                new_state.append(initstate[5])
            # ph0 
            if initstate[8] == -1:
                ph0val = self.init_genotype_freq_sampler() # assume that genotype frequency is the same in all cohorts to the wild population initially.
                new_state.append(ph0val)
            else:
                new_state.append(initstate[6])
            # pc
            if initstate[9] == -1:
                pcval = self.init_genotype_freq_sampler()
                new_state.append(pcval)
            else:
                new_state.append(initstate[7])
        # G & OG
        het_perloci = np.array(pval)[np.arange(0,self.n_genotypes*self.n_locus,self.n_genotypes) + 1]
        if self.discset == -1:
            Gval = np.mean(het_perloci)
        else:
            Gval = self._discretize_idx(np.mean(het_perloci), self.states['G'])
        new_state.append([Gval])
        new_obs.append([Gval])
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
            Nc = np.exp(np.array(self.state)[self.sidx['logNc']]) - 1
            Nc0 = np.exp(np.array(self.state)[self.sidx['logNc0']]) - 1
            q = np.exp(np.array(self.state)[self.sidx["logq"]]) - 1
            p = np.array(self.state)[self.sidx["p"]]
            ph = np.array(self.state)[self.sidx["ph"]]
            ph0 = np.array(self.state)[self.sidx["ph0"]]
            pc = np.array(self.state)[self.sidx["pc"]]
            G = np.array(self.state)[self.sidx["G"]]
            t = np.array(self.state)[self.sidx['t']].astype(int)
            qhat = np.exp(np.array(self.obs)[self.oidx['logqhat']]) - 1
        else:
            N0 = np.array(self.states["N0"])[np.array(self.state)[self.sidx['N0']]]
            N1 = np.array(self.states["N1"])[np.array(self.state)[self.sidx['N1']]]
            Nh = np.array(self.states["Nh"])[np.array(self.state)[self.sidx['Nh']]]
            Nc = np.array(self.states["Nc"])[np.array(self.state)[self.sidx['Nc']]]
            Nc0 = np.array(self.states["Nc0"])[np.array(self.state)[self.sidx['Nc0']]]
            q = np.array(self.states["q"])[np.array(self.state)[self.sidx['q']]]
            p = np.array(self.states['p'])[np.array(self.state)[self.sidx['p']]]
            ph = np.array(self.states['ph'])[np.array(self.state)[self.sidx['ph']]]
            ph0 = np.array(self.states['ph0'])[np.array(self.state)[self.sidx['ph0']]]
            pc = np.array(self.states['pc'])[np.array(self.state)[self.sidx['pc']]]
            G = np.array(self.states['G'])[np.array(self.state)[self.sidx['G']]]
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
                P = (self.alpha*(N0 + self.beta_2*N1))/(1 + self.alpha*(N0 + self.beta_2*N1)/kappa)
                M0 = np.exp(np.random.normal(self.lM0mu, self.lM0sd))
                M1 = np.exp(np.random.normal(self.lM1mu, self.lM1sd))
                summer_mortality = np.exp(-124*M0)*((1 - delfall) + self.tau*deldiff + (1 - self.tau)*self.r0*self.phidiff)
                extra_info['summer_mortality'] = summer_mortality
                N0_next = np.minimum(P*np.exp(-124*M0)*((1 - delfall) + self.tau*deldiff + (1 - self.tau)*self.r0*self.phidiff),np.ones(self.n_reach)*self.N0minmax[1])
                N1_next = np.minimum((N0+N1)*np.exp(-215*M1)*((1-delfall) + self.tau*delfall + (1 - self.tau)*self.r1*self.phifall),np.ones(self.n_reach)*self.N1minmax[1])
                # hatchery stuff
                needed = 2*a/(np.dot(self.b,self.fc)) # amount of f0 needed to produce 'a' fish
                Nh_next = a + 10 # added 10 so that there's no small discrepancy that doesn't allow stocking the produced amount in the fall.
                if np.any(Nc - needed*self.b < 0):
                    newb = self.b.copy()
                    left = a
                    Nc_next = Nc.copy()
                    Nb = np.zeros(self.n_cohorts) # amount of fish used for production from each cohort.
                    while np.any((Nc_next - needed*newb) < 0):
                        diff = Nc_next - needed*newb # difference between Nc and the amount of fish needed fro producing 'a'.
                        negidx = np.where(diff < 0)[0]
                        Nb[negidx] = Nc_next[negidx]
                        Nc_next[negidx] = 0
                        if all(Nc_next == 0):
                            Nh_next = np.sum(Nc/2*self.fc) + 10
                            newb = np.zeros(self.n_cohorts)
                            Nb = Nc.copy()
                            break
                        left = left - np.sum(Nc[negidx]*self.fc[negidx])/2 # divided by 2 because 1:1 male to female is used for production.
                        newb[negidx] = 0
                        newb = newb/np.sum(newb)
                        needed = 2*left/(np.dot(newb,self.fc))
                    Nc_next = Nc_next - needed*newb     
                    Nb += needed*newb
                else:
                    Nc_next = np.maximum(Nc - 2*a/(np.dot(self.b,self.fc))*self.b,0)
                    Nb = needed*self.b # amount of fish used for production from each cohort.
                Nc0_next = np.random.uniform(size=1)*self.eggcollection_max*self.s0egg + np.random.uniform(size=1)*self.larvaecollection_max*self.s0larvae # Nc is the number of females so it's divided by 2. hatchery uses 1:1 sex ratio for production.
                # hydrological stuff. No springflow in fall (stays the same)
                # genetic stuff (reproduction and summer survival impact on allele frequency)
                p_next = []
                ph0_next = []
                ph_next = []
                pc_next = []
                phtensor = np.transpose(ph.reshape(self.n_cohorts, self.n_locus, self.n_genotypes),(1,0,2)) # dim: (self.n_locus, self.n_cohorts, self.n_genotypes)
                for l in range(self.n_locus):
                    # wild  genotype frequency
                    idx1,idx2 = int(l*self.n_genotypes), int((l+1)*self.n_genotypes)
                    gfreq = p[idx1:idx2] # genotype frequency for locus l
                    allelefreq = np.array([gfreq[0]+1/2*gfreq[1],(gfreq[2]+1/2*gfreq[1])]) # allele frequency for locus l
                    pprime = np.array([allelefreq[0]**2,2*allelefreq[0]*allelefreq[1],allelefreq[1]**2]) # genotype frequency of the next generation assuming HW eqbm.
                    Xp = np.random.multinomial(np.sum(N0_next).astype(int),pprime) if np.sum(N0_next) < 1e5 else np.round(np.sum(N0_next)*pprime) # Xp is the number of individuals in the age0 population for each genotype, if the number of alleles (2N0) is larger than 10000, just use the expected values.
                    Yp = np.random.multinomial(np.sum(N1_next).astype(int),gfreq) if np.sum(N1_next) < 1e5 else np.round(np.sum(N1_next)*gfreq) # Yp is the number of individuals in the age1 population for each genotype
                    p_next_perlocus = (Xp + Yp)/np.sum(Xp+Yp) # allele frequency in the population
                    p_next.append(p_next_perlocus)
                    # stock fish genotype frequency
                    if a > 0:
                        #K = np.array([np.random.default_rng().multivariate_hypergeometric(n.astype(int), draw.astype(int)) for  draw,n in zip(np.floor(Nb),np.ceil(phtensor[l]*Nc.reshape(-1,1)))]) # multivariate hypergeometric dist. is multinomial sampling without replacement.
                        K = np.array([np.random.default_rng().multivariate_hypergeometric(n.astype(int), draw.astype(int)) for  draw,n in zip(np.ceil(Nb),np.ceil(phtensor[l]*Nc.reshape(-1,1)))]) # multivariate hypergeometric dist. is multinomial sampling without replacement.
                        K_sum = np.sum(K,axis=1)
                        #K = np.array([np.random.multinomial(int(n), pvals) for n, pvals in zip(Nb, phtensor[l])]) # (cohort, genotype)
                        K_sum[K_sum == 0] = 1 # if the sum of genotype frequency is 0, set it to 1.
                        pc_p = K/(K_sum.reshape(-1,1)) # frequency of selected broodstock for production from each cohort. (cohort, genotype)
                        allelefreq_pc = np.array([pc_p[:,0]+1/2*pc_p[:,1],(pc_p[:,2]+1/2*pc_p[:,1])]) # allele frequency for pc_p
                        pc_pp = np.array([allelefreq_pc[0]**2,2*allelefreq_pc[0]*allelefreq_pc[1],allelefreq_pc[1]**2]).T # genotype frequency of the F1 assuming HW eqbm.
                        Kp = np.array([np.random.multinomial(int(n), pvals) if n < 1e5 else np.round(n*pvals)  for n, pvals in zip(np.round(Nb)*self.fc, pc_pp)]) # (cohort, genotype); genotype frequency of F1 from each cohort selected for breeding. We account for irphi (immediate death after stocking) here in advance.
                        pc_perloci = np.sum(Kp,axis=0)/(np.dot(Nb,self.fc)) # (genotype) genotype frequency of F1.
                        pc_perloci = pc_perloci/np.sum(pc_perloci) # normalize to make sure each locus' genotype frequency sums to 1
                        pc_next.append(pc_perloci)
                    else:
                        pc_next.append(np.zeros(self.n_genotypes))
                    # broodstock genotype frequency
                    Xpprime = np.random.multinomial(int(Nc0_next),pprime) if Nc0_next < 1e5 else np.round(Nc0_next*pprime) # Xpprime is the number individuals in the collected eggs/juv's for each genotype
                    ph0_next_perlocus = Xpprime/np.sum(Xpprime) if np.sum(Xpprime) > 0 else np.zeros(self.n_genotypes) # genotype frequency in the hatchery
                    ph0_next.append(ph0_next_perlocus)
                    Xppprime = np.round(np.ceil(phtensor[l] * Nc.reshape(-1,1)) - K)
                    Xppprime_sum = np.sum(Xppprime,axis=1) # (cohort, 1)
                    Xppprime_sum[Xppprime_sum == 0] = 1 # if the sum of genotype frequency is 0, set it to 1.
                    ph_next_perlocus = Xppprime/(Xppprime_sum.reshape(-1,1)) # (cohort, genotype)
                    ph_next.append(ph_next_perlocus) # (locus, cohort, genotype)
                # reshape ph_next and flatten it.
                ph_next_tensor = np.transpose(np.array(ph_next),(1,0,2))
                ph_next = ph_next_tensor.flatten()
                p_next = np.concatenate(p_next)
                ph0_next = np.concatenate(ph0_next)
                pc_next = np.concatenate(pc_next)
            
                # calculate heterozygosity
                het_perloci = p_next[np.arange(0,self.n_genotypes*self.n_locus,self.n_genotypes) + 1]
                G_next = [np.mean(het_perloci)] # heterozygosity in the population
                # reward & done
                reward = self.extant - self.unitcost*a if a > 0 else self.extant
                done = False
                # update state & obs
                if self.discset == -1:
                    logN0_next = np.log(N0_next+1)
                    logN1_next = np.log(N1_next+1)
                    logNh_next = [np.log(Nh_next+1)]
                    logNc_next = np.log(Nc_next+1)
                    logNc0_next = np.log(Nc0_next+1)
                    OG_next = np.array(self.obs)[self.oidx['OG']] # no change 
                    self.state = list(np.concatenate([logN0_next, logN1_next, logNh_next, logNc_next, logNc0_next, np.array(self.state)[self.sidx["logq"]], p_next, ph_next, ph0_next, pc_next, G_next, t_next]))
                    self.obs = list(np.concatenate([logN0_next, logN1_next, logNh_next, logNc_next, logNc0_next, np.array(self.obs)[self.oidx["logqhat"]], OG_next, t_next]))
                else:
                    N0_next_idx = [self._discretize_idx(val, self.states['N0']) for val in N0_next]
                    N1_next_idx = [self._discretize_idx(val, self.states['N1']) for val in N1_next]
                    Nh_next_idx = [self._discretize_idx(Nh_next, self.states['Nh'])]
                    Nc_next_idx = [self._discretize_idx(val, self.states['Nc']) for val in Nc_next]
                    Nc0_next_idx = [self._discretize_idx(Nc0_next, self.states['Nc0'])]
                    q_next_idx = np.array(self.state)[self.sidx['q']] # no change
                    p_next_idx = self.genotype_freq_discretize(p_next.reshape(self.n_locus, self.n_genotypes))
                    ph_next_idx = np.concatenate([self.genotype_freq_discretize(ph_next_tensor[c]) for c in range(self.n_cohorts)])
                    ph0_next_idx = self.genotype_freq_discretize(ph0_next.reshape(self.n_locus, self.n_genotypes))
                    pc_next_idx = self.genotype_freq_discretize(pc_next.reshape(self.n_locus, self.n_genotypes))
                    G_next_idx = [self._discretize_idx(np.mean(het_perloci), self.states['G'])]
                    OG_next_idx = np.array(self.obs)[self.oidx['G']] # no change 
                    t_next_idx = t_next
                    qhat_next_idx = np.array(self.obs)[self.oidx['qhat']] # no change
                    self.state = list(np.concatenate([N0_next_idx, N1_next_idx, Nh_next_idx, Nc_next_idx, Nc0_next_idx, q_next_idx, p_next_idx, ph_next_idx, ph0_next_idx, pc_next_idx, G_next_idx, t_next_idx]).astype(int))
                    self.obs = list(np.concatenate([N0_next_idx, N1_next_idx, Nh_next_idx, Nc_next_idx, Nc0_next_idx, qhat_next_idx, OG_next_idx, t_next_idx]).astype(int))
            elif t < 3: # fall stocking for angostura or isleta
                # demographic stuff
                if a - Nh <= 1e-7: # valid action choice. stocking is less than the hatchery population size.
                    N0_next = N0.copy()
                    N0_next[t-1] = np.minimum(N0_next[t-1] + a,self.N0minmax[1]) # stocking angostura (t=1) or isleta (t=2) in the fall
                    N1_next = N1.copy()
                    Nh_next = Nh[0] - a
                    Nc0_next = Nc0.copy()
                    Nc_next = Nc.copy()
                    # genetic stuff (stocking effect on allele frequency)
                    # genotype frequency for F1 fish still in the hatchery genotype frequency and F1 fish that's stocked (ps)
                    if a > 0:
                        pc_next = []
                        ps = []
                        for l in range(self.n_locus):
                            idx1,idx2 = int(l*self.n_genotypes), int((l+1)*self.n_genotypes)
                            # pc
                            Kp = np.random.multinomial(Nh_next.astype(int), pc[idx1:idx2]) if Nh_next < 1e5 else np.round(Nh_next*pc[idx1:idx2]) # frequency in the F1 in hatchery after stocking.
                            pc_perlocus = Kp/np.sum(Kp) if Nh_next > 0 else np.zeros(self.n_genotypes)
                            pc_next.append(pc_perlocus)
                            # ps
                            psprime_perlocus = (Nh[0]*pc[idx1:idx2] - Kp)/a # Nh*pc[idx1:idx2]= genotype count before stocking, Kp= genotype count after stocking. subtracting the two gives the genotype count of stocked fish.
                            Kpp = np.random.multinomial(a*self.irphi, psprime_perlocus) if a*self.irphi < 1e5 else np.round(a*self.irphi*psprime_perlocus) # genotype frequency of stocked fish.
                            ps_perlocus = Kpp/np.sum(Kpp) if np.sum(Kpp) > 0 else np.zeros(self.n_genotypes)
                            ps.append(ps_perlocus)
                        pc_next = np.concatenate(pc_next)
                        ps = np.concatenate(ps)
                    else:
                        pc_next = pc.copy()
                        ps = np.zeros(self.n_genotypes*self.n_locus) # no stocking, so ps is zero.
                    # wild genotype frequency
                    p_next = self.p_update_post_stocking(a,p,ps,totN0,totN1) if a > 0 else p.copy()
                    # reward & done
                    reward = self.extant
                    done = False
                else: # invalid action choice that results in heavy penalty with termination
                    N0_next = N1_next = np.zeros(self.n_reach)
                    Nh_next = 0
                    Nc0_next = Nc0.copy()
                    Nc_next = Nc.copy()
                    p_next = np.zeros(self.n_genotypes*self.n_locus)
                    pc_next = np.zeros(self.n_genotypes*self.n_locus)
                    # reward & done
                    reward = 0
                    done = True
                    print("Invalid action choice. Terminating the episode.")
                # hydrological stuff
                q_next = q # no springflow in fall (stays the same)
                qhat_next = qhat # no springflow in fall (stays the same)
                # hatchery genetic stuff.
                ph0_next = ph0.copy() # no change
                ph_next = ph.copy() # no change
                # calculate heterozygosity
                het_perloci = p_next[np.arange(0,self.n_genotypes*self.n_locus,self.n_genotypes) + 1]
                G_next = [np.mean(het_perloci)] # allelic richness in the population
                # update state & obs
                if self.discset == -1:
                    logN0_next = np.log(N0_next+1)
                    logN1_next = np.log(N1_next+1)
                    logNh_next = [np.log(Nh_next+1)]
                    self.state = list(np.concatenate([logN0_next, logN1_next, logNh_next, np.array(self.state)[self.sidx["logNc"]], np.array(self.state)[self.sidx["logNc0"]], np.array(self.state)[self.sidx["logq"]], p_next, ph_next, ph0_next, pc_next, G_next, t_next]))
                    self.obs = list(np.concatenate([logN0_next, logN1_next, logNh_next, np.array(self.obs)[self.oidx["OlogNc"]], np.array(self.obs)[self.oidx["OlogNc0"]],np.array(self.obs)[self.oidx["logqhat"]], np.array(self.obs)[self.oidx["OG"]], t_next]))
                else:
                    N0_next_idx = [self._discretize_idx(val, self.states['N0']) for val in N0_next]
                    N1_next_idx = [self._discretize_idx(val, self.states['N1']) for val in N1_next]
                    Nh_next_idx = [self._discretize_idx(Nh_next, self.states['Nh'])]
                    Nc_next_idx = np.array(self.state)[self.sidx['Nc']] # no change
                    Nc0_next_idx = np.array(self.state)[self.sidx['Nc0']] # no change
                    q_next_idx = np.array(self.state)[self.sidx['q']] # no change
                    p_next_idx = self.genotype_freq_discretize(p_next.reshape(self.n_locus, self.n_genotypes))
                    ph_next_idx = np.array(self.state)[self.sidx['ph']] # no change
                    ph0_next_idx = np.array(self.state)[self.sidx['ph0']] # no change
                    pc_next_idx = self.genotype_freq_discretize(pc_next.reshape(self.n_locus, self.n_genotypes))
                    G_next_idx = [self._discretize_idx(np.mean(het_perloci), self.states['G'])]
                    OG_next_idx = np.array(self.state)[self.sidx['G']] # no change
                    t_next_idx = t_next
                    qhat_next_idx = np.array(self.obs)[self.oidx['qhat']] # no change
                    self.state = list(np.concatenate([N0_next_idx, N1_next_idx, Nh_next_idx, Nc_next_idx, Nc0_next_idx, q_next_idx, p_next_idx, ph_next_idx, ph0_next_idx, pc_next_idx, G_next_idx, t_next_idx]).astype(int))
                    self.obs = list(np.concatenate([N0_next_idx, N1_next_idx, Nh_next_idx, Nc_next_idx, Nc0_next_idx, qhat_next_idx, OG_next_idx, t_next_idx]).astype(int))
            else: # t=3
                if a - Nh <= 1e-7: # valid action choice. stocking is less than the hatchery population size.
                    Mw = np.exp(np.random.normal(self.lMwmu, self.lMwsd))
                    N0_next = N0.copy()
                    N0_next[t-1] = N0_next[t-1] + a # stocking san acacia (t=3) in the fall
                    N0_next = np.minimum(N0_next*np.exp(-150*Mw),np.ones(self.n_reach)*self.N0minmax[1]) # stocking san acacia (t=3) in the fall
                    N1_next = N1*np.exp(-150*Mw)
                    Nh_next = 0
                    Nc_next = np.concatenate([Nc0,Nc[0:-1]*self.sc]) # nc0 becomes age 1 and the rest of the cohorts age by 1 yr, all with some survival rate.
                    Nc0_next = 0 # no age 0 fish left in spring.
                    # genetic stuff (stocking and winter survival impact on allele frequency)
                    ## genotype frequency of stocked fish (ps)
                    if a > 0:
                        ps = []
                        for l in range(self.n_locus):
                            idx1,idx2 = int(l*self.n_genotypes), int((l+1)*self.n_genotypes)
                            Kp = np.random.multinomial(a, pc[idx1:idx2])
                            ps_perlocus = Kp/np.sum(Kp) if np.sum(Kp) > 0 else np.zeros(self.n_genotypes) # genotype frequency of stocked fish.
                            ps.append(ps_perlocus)
                        ps = np.concatenate(ps)
                    else:
                        ps = np.zeros(self.n_genotypes*self.n_locus) # no stocking, so ps is zero.
                    # wild genotype frequency  before winter survival
                    p_prime = self.p_update_post_stocking(a,p,ps,totN0,totN1) if a > 0 else p.copy() 
                    # wild genotype frequency after winter survival
                    p_next = []
                    for l in range(self.n_locus):
                        idx1,idx2 = int(l*self.n_genotypes), int((l+1)*self.n_genotypes)
                        Zp = np.random.multinomial(int(np.sum(N0_next) + np.sum(N1_next)),p_prime[idx1:idx2]) if (np.sum(N0_next) + np.sum(N1_next)) < 1e5 else np.round((np.sum(N0_next) + np.sum(N1_next))*p_prime[idx1:idx2]) # Zp is the number of alleles in the age0 population after winter survival
                        p_next_perlocus = Zp/np.sum(Zp) # allele frequency in the population
                        p_next.append(p_next_perlocus)
                    p_next = np.concatenate(p_next)
                    # reward & done
                    reward = self.extant
                    done = False
                else:
                    N0_next = N1_next = np.zeros(self.n_reach)
                    Nh_next = 0
                    Nc0_next = 0
                    Nc_next = np.concatenate([Nc0,Nc[0:-1]*self.sc]) # nc0 becomes age 1 and the rest of the cohorts age by 1 yr, all with some survival rate.
                    p_next = np.zeros(self.n_genotypes*self.n_locus)
                    # reward & done
                    reward = 0
                    done = True
                    print("Invalid action choice. Terminating the episode.")
                # hydrological stuff
                q_next, qhat_next = self.flowmodel.nextflowNforecast(q) # springflow and forecast in spring
                q_next = q_next[0][0]
                # hatchery genetic stuff.
                ph0_next = np.zeros(self.n_genotypes*self.n_locus) # age0 becomes age 1 and there's no age 0 hatchery fish left
                ph_next = self.ph_winterupdate(ph,ph0,Nc_next)
                pc_next = np.zeros(self.n_genotypes*self.n_locus) # no hatchery fish left in spring.
                # calculate allelic richness 
                het_perloci = p_next[np.arange(0,self.n_genotypes*self.n_locus,self.n_genotypes) + 1]
                G_next = [np.mean(het_perloci)] # allelic richness in the population
                # update state & obs
                if self.discset == -1:
                    logN0_next = np.log(N0_next+1)
                    logN1_next = np.log(N1_next+1)
                    logNh_next = np.array([np.log(Nh_next+1)])
                    logNc_next = np.log(Nc_next+1)
                    logNc0_next = [np.log(Nc0_next+1)]
                    logq_next = np.array([np.log(q_next+1)])
                    logqhat_next = np.array([np.log(qhat_next+1)])
                    self.state = list(np.concatenate([logN0_next, logN1_next, logNh_next, logNc_next, logNc0_next, logq_next, p_next, ph_next, ph0_next, pc_next, G_next, t_next]))
                    self.obs = list(np.concatenate([logN0_next, logN1_next, logNh_next, logNc_next, logNc0_next, logqhat_next, G_next, t_next]))
                else:
                    N0_next_idx = [self._discretize_idx(val, self.states['N0']) for val in N0_next]
                    N1_next_idx = [self._discretize_idx(val, self.states['N1']) for val in N1_next]
                    Nh_next_idx = [self._discretize_idx(Nh_next, self.states['Nh'])]
                    Nc_next_idx = [self._discretize_idx(val, self.states['Nc']) for val in Nc_next]
                    Nc0_next_idx = [self._discretize_idx(Nc0_next, self.states['Nc0'])]
                    q_next_idx = [self._discretize_idx(q_next, self.states['q'])]
                    qhat_next_idx = [self._discretize_idx(qhat_next, self.observations['qhat'])]
                    p_next_idx = self.genotype_freq_discretize(p_next.reshape(self.n_locus, self.n_genotypes))
                    ph_next_idx = np.concatenate([np.array(self.states['ph0'])[np.array(self.state)[self.sidx['ph0']]],np.array(self.states['ph'])[np.array(self.state)[self.sidx['ph'][0:self.n_alleles*(self.n_cohorts-1)]]]])
                    ph0_next_idx = ph0_next.astype(int)
                    G_next_idx = [self._discretize_idx(np.mean(het_perloci), self.states['G'])]
                    t_next_idx = t_next
                    self.state = list(np.concatenate([N0_next_idx, N1_next_idx, Nh_next_idx, Nc_next_idx, Nc0_next_idx, q_next_idx, p_next_idx, ph_next_idx, ph0_next_idx, pc_next_idx, G_next_idx, t_next_idx]).astype(int))
                    self.obs = list(np.concatenate([N0_next_idx, N1_next_idx, Nh_next_idx, Nc_next_idx, Nc0_next_idx, qhat_next_idx, G_next_idx, t_next_idx]).astype(int))
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
                "Nc": list(np.linspace(self.Ncminmax[0], self.Ncminmax[1], 11)), # hatchery population size (1)
                "Nc0": list(np.linspace(self.Nc0minmax[0], self.Nc0minmax[1], 11)), # hatchery population size (1)
                "q": list(np.linspace(self.qminmax[0], self.qminmax[1], 11)), # spring flow in Otowi (1)
                "p": list(np.linspace(self.pminmax[0], self.pminmax[1], 11)), # genotype frequency (n_genotype)             
                "ph": list(np.linspace(self.phminmax[0], self.phminmax[1], 11)), # age1+ hatchery genotype frequency (n_genotype,n_cohorts)
                "ph0": list(np.linspace(self.ph0minmax[0], self.ph0minmax[1], 11)), # age0 hatchery allele frequency  (n_genotype)
                "pc": list(np.linspace(self.pcminmax[0], self.pcminmax[1], 11)), # age0 hatchery allele frequency  (n_genotype)
                "G": list(np.linspace(self.Gminmax[0], self.Gminmax[1], 11)), # heterozygosity (1)
                "t": list(np.arange(self.tminmax[0], self.tminmax[1] + 1, 1)), # time (1)
            }

            observations = {
                "ON0": states['N0'],
                "ON1": states['N1'], 
                "ONh": states['Nh'],
                "ONc": states['Nc'],
                "ONc0": states['Nc0'],
                "qhat": list(np.linspace(self.qhatminmax[0], self.qhatminmax[1] + 1, 11)), # forecasted flow in Otowi dim:(1)
                "OG": states['G'],
                "Ot": states['t'], 
            }
            actions = {
                "a": list(np.linspace(self.aminmax[0], self.aminmax[1], 11)), # stocking/production size (1)
            }
        elif discretization_set == -1: # continuous
            states = {
                "logN0": list(np.log(np.array(self.N0minmax)+1)), # log population size dim:(3)
                "logN1": list(np.log(np.array(self.N1minmax)+1)), # log population size (3)
                "logNh": self.Nhminmax, # hatchery population size (1)
                "logNc": self.Ncminmax, # cohort population size (1)
                "logNc0": self.Nc0minmax, # cohort population size (1)
                "logq": list(np.log(np.array(self.qminmax)+1)), # log spring flow in Otowi (Otowi) (1)
                "p": self.pminmax, # genotype frequency (n_genotype)
                "ph": self.phminmax, # age1+ hatchery genotype frequency (n_genotype,n_cohorts)
                "ph0": self.ph0minmax, # age0 hatchery genotype frequency  (n_genotype)
                "pc": self.pcminmax, # age0 hatchery genotype frequency  (n_genotype)
                "G": self.Gminmax, # heterozygosity (1)
                "t": self.tminmax # time (1)
            }
            observations = {
                "OlogN0": states['logN0'],
                "OlogN1": states['logN1'],
                "OlogNh": states['logNh'],
                "OlogNc": states['logNc'],
                "OlogNc0": states['logNc0'],
                "logqhat": list(np.log(np.array(self.qhatminmax)+1)), # forecasted flow in Otowi dim:(1)
                "OG": states['G'],
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

    def init_genotype_freq_sampler(self):
        """
        set initial genotype frequency such that the heteorzygosity is around where it is now. (0.18)
        the frequency of homozygotes follows the HW eqbm.
        Assume biallelic loci and 2nd index is the heterozygote frequency.
        output:
            (if continuous) genotype frequency: list of length n_genotype (using dirichlet distribution)
            OR
            (if discrete) index of genotype frequency: list of index of genotype frequency
        """
        init_het_freq_expected = 0.18335 # initial heterozygosity frequency as seen in data. average between 2017 and 2018 data
        init_het_freq = np.random.normal(init_het_freq_expected,0.001,size=self.n_locus) 
        init_hom_freq1 = ((1 + np.sqrt(1-2*init_het_freq))/2)**2
        init_hom_freq2 = ((1 - np.sqrt(1-2*init_het_freq))/2)**2
        init_freq_mat = np.array([init_hom_freq1,init_het_freq,init_hom_freq2]).T
        if self.discset == -1: # continuous
            return list(init_freq_mat.flatten())
        else: # discrete
            return self.genotype_freq_discretize(init_freq_mat)
        
    def genotype_freq_discretize(self, freq):
        """
        input:
            freq: list of genotype frequency (continuous) each row is a frequency for a locus
        output:
            index of genotype frequency (discrete)
        Uses Knuth's “rounding to a given sum” trick and runs in O(nlogn) due to one sort.
        """
        scaledfreq = freq*(len(self.states['p']) - 1)
        scaledfreq_flr = np.floor(scaledfreq)
        scaledfrq_flr_margin = np.round(np.sum(scaledfreq_flr,axis=1) - np.sum(scaledfreq,axis=1)).astype(int)
        for i in range(len(scaledfrq_flr_margin)):
            if scaledfrq_flr_margin[i] < 0:
                scaledfrac = scaledfreq[i] - scaledfreq_flr[i]
                scaledfreq_flr[i][np.argsort(scaledfrac)[::-1][0:np.abs(np.round(scaledfrq_flr_margin[i]).astype(int))]] += 1
        return list(scaledfreq_flr.flatten().astype(int))

        #scaledfreq = freq*(len(self.states['p']) - 1)
        #scaledfreq_flr = np.floor(scaledfreq)
        #scaledfrq_flr_margin = round(np.sum(scaledfreq_flr) - np.sum(scaledfreq))
        #if scaledfrq_flr_margin < 0:
        #    scaledfrac = scaledfreq - scaledfreq_flr
        #    scaledfreq_flr[np.argsort(scaledfrac)[::-1][0:np.abs(round(scaledfrq_flr_margin))]] += 1
        #return list(scaledfreq_flr.astype(int))

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
        cpue = np.array([np.random.negative_binomial(self.sampler, p[i], size=sitenum[i])/avg_effort[i]*100 for i in range(len(sitenum))], dtype=object) # cpue is catch per 100square meters.
        # test code below (calculates mean cpue for each reach for different reach population sizes)
        sitenum = np.array([1000, 1000, 1000])
        popsizes = [10e2, 10e3, 10e4, 10e5, 10e6, 10e7]
        mcpue = np.zeros((len(popsizes), self.n_reach))
        for j in range(len(popsizes)):
            popsize = np.ones(3) * popsizes[j]
            avgp = np.mean([self.p0, self.p1])
            avgfallf = (self.fpool_f + self.frun_f)/2 # is the proportion of RGSM in the river segment exposed to sampling
            avgcatch = popsize*avgp*avgfallf*self.thetaf
            p = self.sampler / (avgcatch + self.sampler)
            cpue = np.array([np.random.negative_binomial(self.sampler, p[i], size=sitenum[i])/avg_effort[i]*100 for i in range(len(sitenum))], dtype=object) # cpue is catch per 100square meters.
            mcpue[j] = np.array([np.mean(cpue[i]) for i in range(len(sitenum))], dtype=object) # mean cpue for each reach
        print(self.fpool_f)
        print(self.frun_f)
        return cpue, mcpue
    
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
    
    def p_update_post_stocking(self,a,p,ps,N0sum,N1sum):
        """
        get the next timestep genotype frequency of the wild population after stocking
        input:
            a: stocking amount (1)
            p: genotype frequency (n_genotype,n_locus)
            pc: genotype frequency of F1 (already accounted for stocking death, irphi) (n_genotype,n_locus)
            N0sum: total population size of age 0 fish (1)
            N1sum: total population size of age 1+ fish (1)
        output:
            p_next: next timestep genotype frequency (n_genotype,n_locus)
        """
        p_next = ((N0sum + N1sum)*p + a*self.irphi*ps)/(N0sum+N1sum+a*self.irphi) # (genotype, locus)
        return p_next
            
    def ph_winterupdate(self,ph,ph0,Ncnext):
        """
        get the next timestep genotype frequency of the hatchery population after winter survival
        input:
            ph: hatchery genotype frequency (n_cohorts,n_genotype,n_locus)
            ph0: hatchery genotype frequency of age 0 fish (n_genotype,n_locus)
            Nc: cohort population size after annual survival (1)
            Nc0: cohort population size of age 0 after annual survival (1)
        """
        phtensor = np.transpose(ph.reshape(self.n_cohorts, self.n_locus, self.n_genotypes),(1,0,2)) # dim: (self.n_locus, self.n_cohorts, self.n_genotypes)
        phtensor = phtensor[:,0:-1,:] # remove the last cohort
        ph_next = []
        for l in range(self.n_locus):
            idx1,idx2 = int(l*self.n_genotypes), int((l+1)*self.n_genotypes)
            try:
                Zp = np.array([np.random.multinomial(int(n), pvals) if n < 1e5 else np.round(n*pvals) for n, pvals in zip(Ncnext[1::],phtensor[l])]) # (cohort, genotype)
            except ValueError: # if Ncnext is empty, then Zp will be empty
                doo = 0
            Zp0 = np.random.multinomial(int(Ncnext[0]),ph0[idx1:idx2]) if Ncnext[0] < 1e5 else np.round(Ncnext[0]*ph0[idx1:idx2]) # (genotype)
            Zp = np.vstack((Zp0,Zp)) # (genotype, cohort) 
            Zp_sum = np.sum(Zp,axis=1)
            Zp_sum[Zp_sum == 0] = 1
            ph_perloci = Zp/(Zp_sum.reshape(-1,1))
            ph_next.append(ph_perloci)
        ph_next_tensor = np.transpose(np.array(ph_next),(1,0,2))
        ph_next = ph_next_tensor.flatten()
        return ph_next

    def _compute_mask(self,states=None):
        """Return 1/0 vector of length N_ACTIONS indicating legal choices."""
        if states is None:
            states = np.array(self.obs).reshape(1,-1)
        springidx = states[:,self.oidx['Ot'][0]] == 0
        fallidx = ~springidx
        mask = np.zeros((states.shape[0], len(self.actions['a'])), dtype=int)
        # in spring, you can't produce more than what the broodstock size can produce
        if np.sum(springidx) > 0:
            cohorts = np.atleast_2d(np.exp(states[np.ix_(np.flatnonzero(springidx),self.oidx['OlogNc'])])-1)
            max_production = np.sum((cohorts*self.b*self.fc), axis=1)
            springlegal = max_production[:,None] >= self.actions['a']
            mask[springidx] = springlegal
        if np.sum(fallidx) > 0: # in fall, you cannot stock more than what's in the hatchery.
            hatchery = np.atleast_2d(np.exp(states[np.ix_(np.flatnonzero(fallidx),self.oidx['OlogNh'])]))
            falllegal = hatchery - 1 >= self.actions['a']
            mask[fallidx] = falllegal
        return mask.astype(int)
