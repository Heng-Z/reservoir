import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
import time
from numba import int32, float32    # import the types
from numba.experimental import jitclass



spec = [
    ('value', int32),               # a simple scalar field
    ('array', float32[:]),          # an array field
]

@jitclass(spec)
class Reservoir():
    def __init__(self,M=None,N=1000,p=0.1,g=1.5):
        if M is not None:
            assert M.shape[0] == M.shape[1]
            self.M = M
            self.N = M.shape[0]
        else:
            self.N = N
            self.M = self.sprandn(self.N,self.N,p) *g *1.0/np.sqrt(p*N)
        #connections
        self.Jz = None
        self.Jgi = None
        #states
        self.x = np.random.rand(self.N,1)
        self.r = np.random.rand(self.N,1)

    @staticmethod
    def sprandn(N1,N2,p):
        # import sparse 
        import scipy.sparse as sparse
        # import stats
        import scipy.stats as stats
        rvs = stats.norm(loc=0, scale=1).rvs
        S = sparse.random(N1, N2, density=p, data_rvs=rvs)
        return S.toarray()

    @staticmethod
    def repmat(M,n1,n2=None):
        #copy M (m1,m2) to form multi-dim array of (m1,m2,n1,n2)
        N = np.repeat(M[:,:,np.newaxis], n1, axis=2)
        if n2 is not None:
            N = np.repeat(N[:,:,:,np.newaxis],n2,axis = 3)
        return N

    def get_Jz(self,Nout,pz,fb=1,overlap = None):
        self.Nout = Nout

        num = int(self.N*pz*Nout) - np.mod(int(self.N*pz*Nout),Nout)
        sample_nuerons = np.random.choice(np.arange(self.N),num,replace=False)
        neuro_per_read = int(self.N*pz*Nout)//Nout
        gz = np.zeros((self.N,Nout))
        for j in range(Nout):
            for i in sample_nuerons[j*neuro_per_read:(j+1)*neuro_per_read]:
                gz[i,j] = 1

        self.read_connectivity = gz
        # self.Jz = np.multiply(randn,gz) #(N,Nout)
        self.Jz = np.zeros((self.N,self.Nout))
        self.Jgz = np.multiply(fb*(np.random.rand(self.N,Nout) - 0.5),gz)
        self.read_neurons = sample_nuerons.reshape(Nout,neuro_per_read)
        self.neuro_per_read = neuro_per_read

    def add_input(self,Nin,pi,g):
        self.Nin = Nin
        self.Jgi = g*self.prandn(self.N,self.Nin,pi)

    def internal_train(self,input_series,output_series,dt,aplha,nt,test_input=None):
        '''
        input: input time series of ndarray of dim (Nin,T)
            output time series of ndarray of dim (Nout,T)
        update the training every nt step
        '''
        if input_series is None:
            input_series = np.zeros(output_series.shape)
            self.add_input(output_series.shape[0],0,0)
        assert input_series.shape[1] == output_series.shape[1]
        L = input_series.shape[1]
        Dout = output_series.shape[0]
        #array to record output trajectories during training and testing
        train_out = np.zeros((Dout,L))
        test_out = np.zeros((Dout,L))
        x = self.x
        r = self.r
        Pz = self.repmat((1.0/aplha)*np.eye(self.N),self.Nout)
        P = self.repmat((1.0/aplha)*np.eye(self.N),self.Nout,n2 = self.neuro_per_read)
        #_________________training__________________
        for i in range(L):
            print(i)
            t = dt * i
            x = (1.0 - dt) *x +np.dot(self.M,r*dt) + np.dot(self.Jgi,input_series[:,i]*dt).reshape(-1,1)
            r = np.tanh(x) #(N,1)
            z = np.dot(self.Jz.T,r) # (Nout,1)

            if np.mod(i,nt) ==0:
                for readi in range(self.Nout):
                    w_readi = self.Jz[:,readi]
                    synapse_ind = np.where(w_readi!=0)
                    flt = np.zeros((self.N,self.N))
                    flt[synapse_ind,synapse_ind] = 1

                    Pzi = Pz[:,:,readi]
                    kzi = np.dot(Pzi,np.dot(flt,r)) #(Nout,1)
                    rPr_z = np.dot(r.T,kzi) # scalar
                    c_z = 1.0/(1.0 + rPr_z) # scalar
                    Pz[:,:,readi] = np.dot(Pzi,flt) - np.dot(kzi,(kzi.T*c_z))
                    e_z = z[readi] - output_series[readi,i]

                    dw = -e_z*kzi*c_z
                    self.Jz[:,readi] += dw.reshape(-1,)
                    
                    neurons_read_i = self.read_neurons[readi,:]
                    for idx,neuroni in enumerate(neurons_read_i):
                        w_ni = self.M[neuroni,:].reshape(1,-1) # (1,N)
                        synapse_ind = np.where(w_ni!=0)[1]  # find neurons pre-synaptic to neuron i 
                        flt = np.zeros((self.N,self.N))
                        flt[synapse_ind,synapse_ind] = 1
                        Pi = P[:,:,readi,idx] #(N,N) the actual dim of Pi is num of pre-synapse
                        ki = np.dot(Pi,np.dot(flt,r)) 
                        rPr = np.dot(r.T,ki)
                        c = 1.0/(1.0 + rPr)
                        Pz[:,:,readi] = np.dot(Pi,flt) -np.dot(ki,(ki.T * c))

                        dw = -e_z * ki * c
                        self.M[neuroni,:] += dw.reshape(-1,)
            
            train_out[:,i] = z
        #_________________testing_______________
        if test_input is None:
            test_input = input_series
        L = test_input.shape[1]

        for i in range(L):
            x = (1.0 - dt) *x +np.dot(self.M,r*dt) + np.dot(self.Jgi,input_series[:,i]*dt).reshape(-1,1)
            r = np.tanh(x) #(N,1)
            z = np.dot(self.Jz.T,r) # (Nout,1)

            test_out[:,i] = z
        
        return train_out,test_out

    def fb_train(self,input_series,output_series,dt,aplha,nt,test_input=None,fb=1.0):
        if input_series is None:
            input_series = np.zeros(output_series.shape)
            self.add_input(output_series.shape[0],0,0)
        assert input_series.shape[1] == output_series.shape[1]
        L = input_series.shape[1]
        Nout = output_series.shape[0]
        #array to record output trajectories during training and testing
        train_out = np.zeros((Nout,L))
        test_out = np.zeros((Nout,L))
        weight_train =np.zeros((Nout,L))
        x = self.x
        r = self.r
        z = np.random.randn(Nout,1)
        P_all = self.repmat(np.eye(self.N),Nout)
        flts = []
        for i in range(Nout):
            synapse_ind = np.where(self.read_connectivity[:,i] !=0)
            flt = np.zeros((self.N,self.N))
            flt[synapse_ind,synapse_ind] = 1
            flts.append(flt)
            P_all[:,:,i] = np.dot(P_all[:,:,i],flt)
        for i in range(L):
            # print(i)
            t = dt * i
            # x = (1.0 - dt) *x +np.dot(self.M,r*dt) + np.dot(self.Jgi,input_series[:,i]*dt).reshape(-1,1) + self.Jgz * z *dt
            x = (1.0 - dt) *x +np.dot(self.M,r*dt) + np.dot(self.Jgz ,z *dt)
            r = np.tanh(x) #(N,1)
            z = np.dot(self.Jz.T,r) # (Nout,1)

            if np.mod(i,nt) == 0:
                for readi in range(Nout):
                    P = P_all[:,:,readi]
                    r_p = np.dot(flts[readi],r)
                    k = np.dot(P,r_p) #(N,1)
                    rPr = np.dot(r_p.T,k) # scalar
                    c = 1.0/(1.0 + rPr) # scalar
                    P_all[:,:,readi] = P - np.dot(k,(k.T*c))
                    e = z[readi] - output_series[readi,i]

                    dw = -e*k*c
                    self.Jz[:,readi] += dw.reshape(-1,)
            train_out[:,i] = z.reshape(-1,)
            weight_train[:,i] = np.diag(np.sqrt(np.dot(self.Jz.T,self.Jz)))
        if test_input is None:
            test_input = input_series
        L = test_input.shape[1]

        for i in range(L):
            # x = (1.0 - dt) *x +np.dot(self.M,r*dt) + np.dot(self.Jgi,input_series[:,i]*dt).reshape(-1,1) + self.Jgz *z *dt
            x = (1.0 - dt) *x +np.dot(self.M,r*dt) + np.dot(self.Jgz, z *dt)
            r = np.tanh(x) #(N,1)
            z = np.dot(self.Jz.T,r) # (Nout,1)

            test_out[:,i] = z.reshape(-1,)
        
        return train_out,test_out,weight_train

    def free_run(self,dt,simulation_time):
        x = self.x
        r = self.r
        tspan = np.array(np.arange(0,simulation_time,dt))
        states_T = np.zeros((self.N,len(tspan)))
        for i,t in enumerate(tspan):
            x = (1.0 - dt) *x +np.dot(self.M,r*dt) 
            r = np.tanh(x) #(N,1)
            states_T[:,i] = x[:,0]       
        return states_T

if __name__ == "__main__":
    start = time.time()
    time_sec = 1440
    dt = 0.1
    nt = 2
    alpha = 1.0
    simtime = np.arange(0,time_sec,step=dt).reshape(1,-1)
    simtime2 = np.arange(time_sec,2*time_sec,step=dt).reshape(1,-1)
    amp = 1.3
    freq = 1/60
    ft = np.zeros((2,simtime.shape[1]))
    ft2 = np.zeros((2,simtime2.shape[1]))
    ft[0,:] = ((amp/1.0)*np.sin(1.0*np.pi*freq*simtime) + \
        (amp/2.0)*np.sin(2.0*np.pi*freq*simtime) +  \
        (amp/6.0)*np.sin(3.0*np.pi*freq*simtime) +  \
        (amp/3.0)*np.sin(4.0*np.pi*freq*simtime)).reshape(-1,)

    ft[1,:] = (np.sin(2.0*np.pi*freq*simtime)).reshape(-1,)

    ft2[0,:] = ((amp/1.0)*np.sin(1.0*np.pi*freq*simtime2) + \
        (amp/2.0)*np.sin(2.0*np.pi*freq*simtime2) +  \
        (amp/6.0)*np.sin(3.0*np.pi*freq*simtime2) +  \
        (amp/3.0)*np.sin(4.0*np.pi*freq*simtime2)).reshape(-1,)
    ft2[1,:] = (np.sin(2.0*np.pi*freq*simtime2)).reshape(-1,)
    nn = Reservoir(N=1000,p=0.5,g=1.5)
    nn.get_Jz(2,0.2,fb=1.0) #(Nout,pz,g,fb=1)
    [train_out,test_out,weight_train] = nn.fb_train(None,ft,dt,alpha,nt,fb=1.0) #(input_series,output_series,dt,aplha,nt,test_input=None)
    end = time.time()
    print('***time consume:****',start-end)
    plt.figure()
    plt.subplot(3,1,1)
    plt.plot(simtime.T,ft.T,'b',simtime.T,train_out.T,'r')
    plt.title('training')
    plt.subplot(3,1,2)
    plt.plot(simtime.T,ft2.T,'b',simtime.T,test_out.T,'g')
    plt.title('testing')
    plt.subplot(3,1,3)
    plt.plot(simtime.T,weight_train.T)
    plt.title('weight')
    # plt.figure()
    # plt.subplot(2,1,1)
    # plt.plot(simtime.T,ft2.T,'b')
    # plt.subplot(2,1,2)
    # plt.plot(simtime.T,test_out.T,'g')
    plt.show()
    

                        





    

        
