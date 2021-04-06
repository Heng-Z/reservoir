import numpy as np
import scipy.sparse as sparse
import scipy.stats as stats
def conv_gauss(sig,sigma=2):
    rg = np.arange(-3*sigma,3*sigma+1,1)
    ker = np.exp(-(rg/sigma)**2/2)
    conved = np.convolve(sig.reshape(-1,),ker,mode='same')/5
    return conved

def sprandn(N1,N2,p):
    rvs = stats.norm(loc=0, scale=1).rvs
    S = sparse.random(N1, N2, density=p, data_rvs=rvs)
    return S.toarray()