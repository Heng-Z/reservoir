import numpy as np
from numpy import matlib
def sprandn(N1,N2,p):
    # import sparse 
    import scipy.sparse as sparse
    # import stats
    import scipy.stats as stats
    rvs = stats.norm(loc=0, scale=1).rvs
    S = sparse.random(N1, N2, density=p, data_rvs=rvs)
    return S.toarray()

def repmat(M,n1,n2=None):
    #copy M (m1,m2) to form multi-dim array of (m1,m2,n1,n2)
    N = np.repeat(M[:,:,np.newaxis], n1, axis=2)
    if n2 is not None:
        N = np.repeat(N[:,:,:,np.newaxis],n2,axis = 3)
    return N

class Reservoir():
    def __init__(self,M=None,N=1000,p=0.1,g=1.5):
        if M is not None:
            assert M.shape[0] == M.shape[1]
            self.M = M
            self.N = M.shape[0]
        else:
            self.N = N
            self.M = sprandn(self.N,self.N,p) *g *1.0/np.sqrt(p*N)
        self.Jz = None
        self.Jgi = None

    def get_Jz(self,Nout,pz,g,overlap = None):
        self.Nout = Nout

        num = int(self.N*pz*Nout) - np.mod(int(self.N*pz*Nout),Nout)
        sample_nuerons = np.random.choice(np.arange(self.N),num)
        gz = np.zeros((self.N,Nout))
        for j in range(Nout):
            for i in sample_nuerons[j*Nout:(j+1)*Nout]:
                gz[i,j] = 1

        randn = g*np.random.normal(size =(self.N,Nout))
        self.Jz = np.multiply(randn,gz)
        self.read_neurons = sample_nuerons.reshape(Nout,-1)
        self.neuro_per_read = self.read_neurons.shape[1]

    def add_input(self,Nin,pi,g):
        self.Nin = Nin
        self.Jgi = g*sprandn(self.N,self.Nin,pi)

    def train(self,input_series,output_series,dt,aplha,nt,test_input=None):
        '''
        input: input time series of ndarray of dim (Nin,T)
            output time series of ndarray of dim (Nout,T)
        update the training every nt step
        '''
        if input_series is None:
            input_series = np.zeros(output_series.shape)
        assert input_series.shape[1] == output_series.shape[1]
        L = input_series.shape[1]
        Dout = output_series.shape[1]
        #array to record output trajectories during training and testing
        train_out = np.zeros((Dout,L))
        test_out = np.zeros((Dout,L))
        x = np.random.rand(self.N)
        r = np.random.rand(self.N)
        Pz = repmat((1.0/aplha)*np.eye(self.neuro_per_read),self.Nout)
        P = repmat((1.0/aplha)*np.eye(self.N),self.Nout,n2 = self.neuro_per_read)
        #_________________training__________________
        for i in range(L):
            t = dt * i
            x = (1.0 - dt) *x +np.dot(self.M,r*dt) + np.dot(self.Jgi,input_series[:,i]*dt)
            r = np.tanh(x) #(N,1)
            z = np.dot(self.Jz.T,r) # (Nout,1)

            if np.mod(i,nt) ==0:
                for readi in range(self.Nout):
                    Pzi = Pz[:,:,readi]
                    kzi = np.dot(Pzi,r) #(Nout,1)
                    rPr_z = np.dot(r.T,kzi) # scalar
                    c_z = 1.0/(1.0 + rPr_z) # scalar
                    Pz[:,:,readi] = Pzi - np.dot(kzi,(kzi.T*c_z))
                    e_z = z[readi] - output_series[readi,i]

                    dw = -e_z*kzi*c
                    self.Jz[:,readi] += dw
                    
                    neurons_read_i = self.read_neurons[readi,:]
                    for neuroni in neurons_read_i:
                        w_ni = self.M[neuroni,:] # (1,N)
                        synapse_ind = np.where(w_ni!=0)[1]  # find neurons pre-synaptic to neuron i 
                        flt = np.zeros(self.N,self.N)
                        flt[synapse_ind,synapse_ind] = 1
                        Pi = P[:,:,readi,neuroni] #(N,N) the actual dim of Pi is num of pre-synapse
                        ki = np.dot(Pi,np.dot(flt,r)) 
                        rPr = np.dot(r.T,ki)
                        c = 1.0/(1.0 + rPr)
                        Pz[:,:,readi] = np.dot(Pi,flt) -np.dot(ki,(ki.T * c))

                        dw = -e_z * ki * c
                        self.M[neuroni,:] += dw
            
            train_out[:,i] = z
        #_________________testing_______________
        if test_input is None:
            test_input = input_series
        L = test_input.shape[1]

        for i in range(L):
            x = (1.0 - dt) *x +np.dot(self.M,r*dt) + np.dot(self.Jgi,input_series[:,i]*dt)
            r = np.tanh(x) #(N,1)
            z = np.dot(self.Jz.T,r) # (Nout,1)

            test_out[:,i] = z
        
        return train_out,test_out

if __name__ == "__main__":
    time_sec = 100 
    dt = 0.1
    simtime = np.arange(0,time_sec,step=dt)
    amp = 1.3;
    freq = 1/60;
    ft = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime) + \
        (amp/2.0)*np.sin(2.0*np.pi*freq*simtime) +  \
        (amp/6.0)*np.sin(3.0*np.pi*freq*simtime) +  \
        (amp/3.0)*np.sin(4.0*np.pi*freq*simtime)

    nn = Reservoir(N=1000,p=0.3,g=1.5)
    nn.get_Jz(1,0.3,1)
    [train_out,test_out] = nn.train(None,ft,dt,1,2)


                        





    

        
