import numpy as np
import cupy as cp
from numpy import matlib
import matplotlib.pyplot as plt
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
    N = cp.repeat(M[:,:,cp.newaxis], n1, axis=2)
    if n2 is not None:
        N = cp.repeat(N[:,:,:,cp.newaxis],n2,axis = 3)
    return N

class Reservoir():
    def __init__(self,M=None,N=1000,p=0.1,g=1.5):
        if M is not None:
            assert M.shape[0] == M.shape[1]
            self.M = M
            self.N = M.shape[0]
        else:
            self.N = N
            self.M = sprandn(self.N,self.N,p) *g *1.0/cp.sqrt(p*N)
        #connections
        self.Jz = None
        self.Jgi = None
        #states
        self.x = cp.random.rand(self.N,1)
        self.r = cp.random.rand(self.N,1)



    def get_Jz(self,Nout,pz,g,overlap = None):
        self.Nout = Nout

        num = int(self.N*pz*Nout) - cp.mod(int(self.N*pz*Nout),Nout)
        sample_nuerons = cp.random.choice(cp.arange(self.N),num,replace=False)
        neuro_per_read = int(self.N*pz*Nout)//Nout
        gz = cp.zeros((self.N,Nout))
        for j in range(Nout):
            for i in sample_nuerons[j*neuro_per_read:(j+1)*neuro_per_read]:
                gz[i,j] = 1

        randn = g*cp.random.normal(size =(self.N,Nout))
        self.read_connectivity = gz
        # self.Jz = cp.multiply(randn,gz) #(N,Nout)
        self.Jz = cp.zeros((self.N,self.Nout))
        self.read_neurons = sample_nuerons.reshape(Nout,neuro_per_read)
        self.neuro_per_read = neuro_per_read

    def add_input(self,Nin,pi,g):
        self.Nin = Nin
        self.Jgi = g*sprandn(self.N,self.Nin,pi)

    def internal_train(self,input_series,output_series,dt,aplha,nt,test_input=None):
        '''
        input: input time series of ndarray of dim (Nin,T)
            output time series of ndarray of dim (Nout,T)
        update the training every nt step
        '''
        if input_series is None:
            input_series = cp.zeros(output_series.shape)
            self.add_input(output_series.shape[0],0,0)
        assert input_series.shape[1] == output_series.shape[1]
        L = input_series.shape[1]
        Dout = output_series.shape[0]
        #array to record output trajectories during training and testing
        train_out = cp.zeros((Dout,L))
        test_out = cp.zeros((Dout,L))
        x = self.x
        r = self.r
        Pz = repmat((1.0/aplha)*cp.eye(self.N),self.Nout)
        P = repmat((1.0/aplha)*cp.eye(self.N),self.Nout,n2 = self.neuro_per_read)
        #_________________training__________________
        for i in range(L):
            print(i)
            t = dt * i
            x = (1.0 - dt) *x +cp.dot(self.M,r*dt) + cp.dot(self.Jgi,input_series[:,i]*dt).reshape(-1,1)
            r = cp.tanh(x) #(N,1)
            z = cp.dot(self.Jz.T,r) # (Nout,1)

            if cp.mod(i,nt) ==0:
                for readi in range(self.Nout):
                    w_readi = self.Jz[:,readi]
                    synapse_ind = cp.where(w_readi!=0)
                    flt = cp.zeros((self.N,self.N))
                    flt[synapse_ind,synapse_ind] = 1

                    Pzi = Pz[:,:,readi]
                    kzi = cp.dot(Pzi,cp.dot(flt,r)) #(Nout,1)
                    rPr_z = cp.dot(r.T,kzi) # scalar
                    c_z = 1.0/(1.0 + rPr_z) # scalar
                    Pz[:,:,readi] = cp.dot(Pzi,flt) - cp.dot(kzi,(kzi.T*c_z))
                    e_z = z[readi] - output_series[readi,i]

                    dw = -e_z*kzi*c_z
                    self.Jz[:,readi] += dw.reshape(-1,)
                    
                    neurons_read_i = self.read_neurons[readi,:]
                    for idx,neuroni in enumerate(neurons_read_i):
                        w_ni = self.M[neuroni,:].reshape(1,-1) # (1,N)
                        synapse_ind = cp.where(w_ni!=0)[1]  # find neurons pre-synaptic to neuron i 
                        flt = cp.zeros((self.N,self.N))
                        flt[synapse_ind,synapse_ind] = 1
                        Pi = P[:,:,readi,idx] #(N,N) the actual dim of Pi is num of pre-synapse
                        ki = cp.dot(Pi,cp.dot(flt,r)) 
                        rPr = cp.dot(r.T,ki)
                        c = 1.0/(1.0 + rPr)
                        Pz[:,:,readi] = cp.dot(Pi,flt) -cp.dot(ki,(ki.T * c))

                        dw = -e_z * ki * c
                        self.M[neuroni,:] += dw.reshape(-1,)
            
            train_out[:,i] = z
        #_________________testing_______________
        if test_input is None:
            test_input = input_series
        L = test_input.shape[1]

        for i in range(L):
            x = (1.0 - dt) *x +cp.dot(self.M,r*dt) + cp.dot(self.Jgi,input_series[:,i]*dt).reshape(-1,1)
            r = cp.tanh(x) #(N,1)
            z = cp.dot(self.Jz.T,r) # (Nout,1)

            test_out[:,i] = z
        
        return train_out,test_out

    def fb_train(self,input_series,output_series,dt,aplha,nt,test_input=None,fb=1.0):
        if input_series is None:
            input_series = cp.zeros(output_series.shape)
            self.add_input(output_series.shape[0],0,0)
        assert input_series.shape[1] == output_series.shape[1]
        L = input_series.shape[1]
        # Dout = output_series.shape[0]
        #array to record output trajectories during training and testing
        train_out = cp.zeros((1,L))
        test_out = cp.zeros((1,L))
        weight_train =cp.zeros((1,L))
        x = self.x
        r = self.r
        z = cp.random.randn(1)
        P = cp.eye(self.N)
        self.Jgz = fb*(cp.random.rand(self.N,1) -0.5)
        
        w_i = self.Jz #(N,1)
        synapse_ind = cp.where(self.read_connectivity[:,0] !=0)
        flt = cp.zeros((self.N,self.N))
        flt[synapse_ind,synapse_ind] = 1
        P = cp.dot(P,flt)
        for i in range(L):
            print(i)
            t = dt * i
            # x = (1.0 - dt) *x +cp.dot(self.M,r*dt) + cp.dot(self.Jgi,input_series[:,i]*dt).reshape(-1,1) + self.Jgz * z *dt
            x = (1.0 - dt) *x +cp.dot(self.M,r*dt) + self.Jgz * z *dt
            r = cp.tanh(x) #(N,1)
            z = cp.dot(self.Jz.T,r) # (Nout,1)

            if cp.mod(i,nt) == 0:
                r_p = cp.dot(flt,r)
                k = cp.dot(P,r_p) #(N,1)
                rPr = cp.dot(r_p.T,k) # scalar
                c = 1.0/(1.0 + rPr) # scalar
                P = P - cp.dot(k,(k.T*c))
                e = z - output_series[0,i]

                dw = -e*k*c
                self.Jz += dw.reshape(-1,1)
            train_out[0,i] = z
            weight_train[0,i] = cp.sqrt(cp.dot(self.Jz.T,self.Jz))
        if test_input is None:
            test_input = input_series
        L = test_input.shape[1]

        for i in range(L):
            # x = (1.0 - dt) *x +cp.dot(self.M,r*dt) + cp.dot(self.Jgi,input_series[:,i]*dt).reshape(-1,1) + self.Jgz *z *dt
            x = (1.0 - dt) *x +cp.dot(self.M,r*dt) + self.Jgz * z *dt
            r = cp.tanh(x) #(N,1)
            z = cp.dot(self.Jz.T,r) # (Nout,1)

            test_out[0,i] = z
        
        return train_out,test_out,weight_train

            





    def free_run(self,dt,simulation_time):
        x = self.x
        r = self.r
        tspan = cp.array(cp.arange(0,simulation_time,dt))
        states_T = cp.zeros((self.N,len(tspan)))
        for i,t in enumerate(tspan):
            x = (1.0 - dt) *x +cp.dot(self.M,r*dt) 
            r = cp.tanh(x) #(N,1)
            states_T[:,i] = x[:,0]       
        return states_T

    
            

if __name__ == "__main__":
    time_sec = 1440
    dt = 0.1
    nt = 2
    alpha = 1.0
    simtime = cp.arange(0,time_sec,step=dt).reshape(1,-1)
    simtime2 = cp.arange(time_sec,2*time_sec,step=dt).reshape(1,-1)
    amp = 1.3
    freq = 1/60
    ft = (amp/1.0)*cp.sin(1.0*cp.pi*freq*simtime) + \
        (amp/2.0)*cp.sin(2.0*cp.pi*freq*simtime) +  \
        (amp/6.0)*cp.sin(3.0*cp.pi*freq*simtime) +  \
        (amp/3.0)*cp.sin(4.0*cp.pi*freq*simtime)
    ft = ft.reshape(1,-1)
    ft2 = (amp/1.0)*cp.sin(1.0*cp.pi*freq*simtime2) + \
        (amp/2.0)*cp.sin(2.0*cp.pi*freq*simtime2) +  \
        (amp/6.0)*cp.sin(3.0*cp.pi*freq*simtime2) +  \
        (amp/3.0)*cp.sin(4.0*cp.pi*freq*simtime2)
    ft2 = ft2.reshape(1,-1)
    nn = Reservoir(N=1000,p=0.5,g=1.5)
    nn.get_Jz(1,0.2,1) #(Nout,pz,g)
    [train_out,test_out,weight_train] = nn.fb_train(None,ft,dt,alpha,nt,fb=1.0) #(input_series,output_series,dt,aplha,nt,test_input=None)
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
    

                        





    

        
