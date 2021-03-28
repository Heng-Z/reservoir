import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
import cupy as cp
import time
import datetime
from tqdm import tqdm
def sprandn(N1,N2,p):
    # import sparse 
    import scipy.sparse as sparse
    # import stats
    import scipy.stats as stats
    rvs = stats.norm(loc=0, scale=1).rvs
    S = sparse.random(N1, N2, density=p, data_rvs=rvs)
    return cp.asarray(S.toarray())

class WormNet:
    def __init__(self,N):
        #set parameters
        self.N = N
        self.inhb_ratio = 0.3
        self.init_type_vector() # init self.type_vec self.inhb_idx
        self.M = self.init_connectome()
        # set state
        self.v = cp.random.rand(self.N,1)*40-40
        self.s = cp.random.rand(self.N,1)
        self.Ecell = -40 + cp.random.rand(self.N,1)*10 
        self.Gc = 7 + cp.random.rand(self.N,1) *6
        self.C = abs(cp.random.randn(self.N,1)/10 +1)
        self.ar = 1
        self.ad = 5
        self.beta = 0.125
        self.vth = self.decide_vth()

    def init_type_vector(self):
        vec = cp.ones(self.N)
        self.inhb_idx = cp.random.choice(cp.arange(self.N),int(self.N*self.inhb_ratio),replace=overlap)
        vec[self.inhb_idx] = -1
        self.type_vector = vec       

    def init_connectome(self):
        pass

    def decide_vth(self):
        return -35
    def run(self,inputs=None,dt = 0.1):
        v = self.v
        s = self.s
        v_run = cp.zeros((self.N,L))
        s_run = cp.zeros((self.N,L))

        for i in range(L):
            Igap = cp.dot(self.Gg,cp.ones(self.N,1))*v -cp.dot(self.Gg,v)
            Isyn = cp.dot(self.Gs,s)*v - cp.dot(self.Gs,s*self.E)
            v = v +(-cp.multiply(self.Gc,v-self.Ecell) - Igap - Isyn + inputs[:,i].reshape(-1,1))*dt/self.Cap
            s =s + ( self.ar *(1/cp.exp(-self.beta*(v-self.Vth))*(1-s) - self.ad * s) *dt
            v_run(:,i:i+1) = v
            s_run(:,i:i+1) = s

        return v_run, s_run

    
        

