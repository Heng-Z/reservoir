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

def repmat(M,n1,n2=None):
    #copy M (m1,m2) to form multi-dim array of (m1,m2,n1,n2)
    N = cp.repeat(M[cp.newaxis,:,:], n1, axis=0)
    if n2 is not None:
        N = cp.repeat(N[cp.newaxis,:,:,:],n2,axis = 0)
    return N

def update_weight(P_all,flts,r,z,Jz,target,delta=0):
    for readi in range(P_all.shape[0]):
        P = P_all[readi]
        r_p = cp.dot(flts[readi],r)
        k = cp.dot(P,r_p) #(N,1)
        rPr = cp.dot(r_p.T,k) # scalar
        c = 1.0/(1.0 + (1+delta)/(1-delta)*rPr) # scalar
        P_all[readi] = P/(1-delta) - (1+delta)/(1-delta)**2* cp.dot(k,(k.T*c))
        # P_all[readi] = P - cp.dot(k,(k.T*c))
        e = z[readi] - target[readi]
        dw = -e*k*c *(1+delta)/(1-delta)
        Jz[:,readi] += dw.reshape(-1,)
    return P_all,Jz
    
def update_readout_synp(P_all,flts,r,z,Jz,M,target):
    for readi in range(self.Nout):
        P = P_all[readi]
        r_p = cp.dot(flts[readi],r)
        k = cp.dot(P,r_p) #(N,1)
        rPr = cp.dot(r_p.T,k) # scalar
        c = 1.0/(1.0 + rPr) # scalar
        P_all[readi] = P - cp.dot(k,(k.T*c))
        e = z[readi] - target[readi]
        dw = -e*k*c
        Jz[:,readi] += dw.reshape(-1,)
        
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
def update_one(P,flt,r,e,J,delta=0):
    #add discount factor delta to perform weighted RLS
    r_p = cp.multiply(flt.reshape(-1,1),r)
    k = cp.dot(P,r_p) #(N,1)
    rPr = cp.dot(r_p.T,k) # scalar
    c = 1.0/(1.0 + (1+delta)/(1-delta)*rPr) # scalar
    P = P/(1-delta) - (1+delta)/(1-delta)**2 * cp.dot(k,(k.T*c))
    dw = -e*k*c*(1+delta)/(1-delta)
    J += dw.reshape(-1,)
    return P,J

class Reservoir():
    def __init__(self,M=None,N=1000,p=0.1,g=1.6):
        if M is not None:
            assert M.shape[0] == M.shape[1]
            self.M = M
            self.N = M.shape[0]
        else:
            self.N = N
            self.M = sprandn(self.N,self.N,p) *g *1.0/cp.sqrt(p*N)
        #connections
        self.inter_connectivity = (abs(self.M)>1e-6).astype(np.uint8)
        self.Jz = None
        self.Jgi = None
        #states
        self.x = cp.random.rand(self.N,1)
        self.r = cp.random.rand(self.N,1)
        self.time_coef = cp.random.randn(N,1)/5 +1
        self.discount = 0

    def change_time_coef(self):
        self.time_coef[0:self.N//3] = cp.random.randn(self.N//3,1)*2 + 10
        self.time_coef[self.N//3:self.N//3*2] =  10*cp.random.randn(self.N//3,1) + 100

    def get_Jz(self,Nout,pz,fb=1,overlap = False):
        self.Nout = Nout

        num = int(self.N*pz*Nout) - cp.mod(int(self.N*pz*Nout),Nout)

        sample_nuerons = cp.random.choice(cp.arange(self.N),int(num),replace=overlap)
        neuro_per_read = int(self.N*pz)
        gz = cp.zeros((self.N,Nout))
        for j in range(Nout):
            for i in sample_nuerons[j*neuro_per_read:(j+1)*neuro_per_read]:
                gz[i,j] = 1

        self.read_connectivity = gz
        # self.Jz = cp.multiply(randn,gz) #(N,Nout)
        self.Jz = cp.zeros((self.N,self.Nout))
        self.Jgz = cp.multiply(fb*(cp.random.rand(self.N,Nout) - 0.5),gz)
        self.read_neurons = sample_nuerons.reshape(Nout,neuro_per_read)
        self.neuro_per_read = neuro_per_read

    def add_input(self,Nin,pi,g):
        self.Nin = Nin
        self.Jgi = g*sprandn(self.N,self.Nin,pi)

    def internal_train(self,input_series,output_series,dt,alpha,nt,nl =0,test_input=None):
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
        weight_train =cp.zeros((self.Nout,L))
        noise = cp.random.randn(self.N,L)*nl
        x = self.x
        r = self.r
        z = cp.random.randn(self.Nout,1)
        P_z = repmat((1.0/alpha)*cp.eye(self.N),self.Nout)
        P_syn = repmat((1.0/alpha)*cp.eye(self.N),self.neuro_per_read,n2 = self.Nout)
        #_________initiate inverse Correlation Matrices and filters_____________
        for i in range(self.Nout):
            flt = self.read_connectivity[:,i]
            P_z[i] = np.multiply(P_z[i],flt*flt.T)
            for j,idx in enumerate(self.read_neurons[:,i]):
                flt = self.M[idx,:].T
                P_syn[i,j] = np.multiply(P_z[i],flt*flt.T)


        #_________________training__________________
        for i in tqdm(range(L)):
            # print(i)
            t = dt * i
            # x = (1.0 - dt) *x +cp.dot(self.M,r*dt) + cp.dot(self.Jgi,input_series[:,i]*dt).reshape(-1,1)
            x = (1.0 - dt) *x +cp.dot(self.M,r*dt) + cp.dot(self.Jgz ,z *dt) + noise[:,i].reshape(-1,1) *dt + cp.dot(self.Jgi,input_series[:,i].reshape(-1,1)*dt)
        
            r = cp.tanh(x) #(N,1)
            z = cp.dot(self.Jz.T,r) # (Nout,1)

            if cp.mod(i,nt) ==0:
                for readi in range(self.Nout):
                    e_z = z[readi] - output_series[readi,i]
                    [P_z[readi],self.Jz[:,readi]] = update_one(P_z[readi],self.read_connectivity[:,readi],r,e_z,self.Jz[:,readi])

                    neurons_read_i = self.read_neurons[readi,:]
                    for idx,neuroni in enumerate(neurons_read_i):
                        [P_syn[readi,neuroni], new_raw] = update_one(P_syn[readi,neuroni],self.inter_connectivity[neuroni,:].T,r,e_z,self.M[neuroni].T)
                        self.M[neuroni] = new_raw.T
            
            train_out[:,i] = z.reshape(-1,)
            weight_train[:,i] = cp.diag(cp.sqrt(cp.dot(self.Jz.T,self.Jz)))
        #_________________testing_______________
        if test_input is None:
            test_input = input_series
        L = test_input.shape[1]

        for i in range(L):
            x = (1.0 - dt) *x +cp.dot(self.M,r*dt) + cp.dot(self.Jgz ,z *dt) + noise[:,i].reshape(-1,1) *dt + cp.dot(self.Jgi,test_input[:,i].reshape(-1,1)*dt)
            r = cp.tanh(x) #(N,1)
            z = cp.dot(self.Jz.T,r) # (Nout,1)
            # print(r.shape,x.shape,z.shape)
            test_out[:,i] = z.reshape(-1,)
        
        return train_out,test_out,weight_train

    def fb_train(self,input_series,output_series,dt,alpha,nt,nl =0,test_input=None):
        if input_series is None:
            input_series = cp.zeros(output_series.shape)
            self.add_input(output_series.shape[0],0,0)
        if test_input is None:
            test_input = input_series
        assert input_series.shape[1] == output_series.shape[1]
        L = input_series.shape[1]
        Nout = output_series.shape[0]
        #array to record output trajectories during training and testing
        train_out = cp.zeros((Nout,L))
        test_out = cp.zeros((Nout,L))
        weight_train =cp.zeros((Nout,L))
        x = self.x
        r = self.r
        z = cp.random.randn(Nout,1)
        P_all = repmat(cp.eye(self.N)/alpha,Nout)
        flts = []
        noise = cp.random.randn(self.N,L)*nl
        noise2 = cp.random.randn(self.N,L)*nl
        for i in range(Nout):
            synapse_ind = cp.where(self.read_connectivity[:,i] !=0)
            flt = cp.zeros((self.N,self.N))
            flt[synapse_ind[0],synapse_ind[0]] = 1
            flts.append(flt)
            P_all[i] = cp.dot(P_all[i],flt)
        for i in tqdm(range(L)):
            # print(i)
            t = dt * i
            # x = (1.0 - dt) *x +cp.dot(self.M,r*dt) + cp.dot(self.Jgi,input_series[:,i]*dt).reshape(-1,1) + self.Jgz * z *dt

            #coef * (x'-x)/dt = -x + Mx --> x' = x+dt/coef *(-x+MX)
            # x = (1.0 - dt) *x +cp.dot(self.M,r*dt) + cp.dot(self.Jgz ,z*dt) + noise[:,i].reshape(-1,1) *dt + cp.dot(self.Jgi,input_series[:,i].reshape(-1,1)*dt)
            
            #when consider different time scale neurons:
            x =x+ (-x +cp.dot(self.M,r) + cp.dot(self.Jgz ,z) + noise[:,i].reshape(-1,1) + cp.dot(self.Jgi,input_series[:,i].reshape(-1,1)))*dt/self.time_coef

            r = cp.tanh(x) #(N,1)
            z = cp.dot(self.Jz.T,r) # (Nout,1)
            if cp.mod(i,nt) == 0:
            #____________UPDATE PARAMETERS_________
                [P_all, self.Jz] = update_weight(P_all,flts,r,z,self.Jz,output_series[:,i],delta=self.discount)
            train_out[:,i] = z.reshape(-1,)
            weight_train[:,i] = cp.diag(cp.sqrt(cp.dot(self.Jz.T,self.Jz)))
        if test_input is None:
            test_input = input_series
        L = test_input.shape[1]
        # print(P)
        # print(flts[1])
        for i in range(L):
            # x = (1.0 - dt) *x +cp.dot(self.M,r*dt) + cp.dot(self.Jgi,input_series[:,i]*dt).reshape(-1,1) + self.Jgz *z *dt
            # x = (1.0 - dt) *x +cp.dot(self.M,r*dt) + cp.dot(self.Jgz, z *dt) + noise2[:,i].reshape(-1,1) *dt + cp.dot(self.Jgi,test_input[:,i].reshape(-1,1)*dt)

            x =x+ (-x +cp.dot(self.M,r) + cp.dot(self.Jgz ,z) + noise[:,i].reshape(-1,1) + cp.dot(self.Jgi,test_input[:,i].reshape(-1,1)))*dt/self.time_coef

            r = cp.tanh(x) #(N,1)
            z = cp.dot(self.Jz.T,r) # (Nout,1)

            test_out[:,i] = z.reshape(-1,)
        
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

def triangle_wave(simtime):
    #simtime is in shape (1,T)
    out = cp.zeros(simtime.shape)
    a = simtime.shape[1]//8
    b = simtime.shape[1]//20
    for i in range(8):
        out[0,a*i:a*i+b] = cp.array(cp.arange(b))/float(b)
    return out

def norm_percentile(signals,p1=5,p2=95,pcnt=True):
    # n signals array are of dim (n,T)
    out = cp.zeros(signals.shape)
    for i in range(signals.shape[0]):
        s = signals[i,:]
        if pcnt == True:
            [n,m] = cp.percentile(s,[p1,p2])
            out[i,:] = (s - n)/(m-n)
        else:
            out[i,:] = (s-cp.mean(s))/cp.std(s)
    return out
def norm_weight(signals,weights):
    out = cp.zeros(signals.shape)
    for i in range(signals.shape[0]):
        s = signals[i,:]
        out[i,:] = (s - cp.mean(s))/weights[i]
    return out
if __name__ == "__main__":
    start = time.time()
    ct = datetime.datetime.now().strftime("%H_%M_%S")
    time_sec = 500
    dt = 0.1
    nt = 3
    N = 500
    for i in range(50):
        # alpha = 1+float(cp.random.rand(1))
        alpha = 0.6
        # fb =1.5+float(cp.random.rand(1))
        fb = 2.2
        g = 1.6
        Pz = 0.11
        # Pgg = float(cp.random.rand(1))*0.75
        Pgg = 0.01
        nn = Reservoir(N=N,p=Pgg,g=g)
        nn.get_Jz(4,Pz,fb=fb) #(Nout,pz,fb=1)
        simtime = cp.arange(0,time_sec,step=dt).reshape(1,-1)
        simtime2 = cp.arange(time_sec,2*time_sec,step=dt).reshape(1,-1)
        amp = 1.3
        freq = 1/20
        ft = cp.zeros((4,simtime.shape[1]))
        ft2 = cp.zeros((4,simtime2.shape[1]))
        ft[0,:] = ((amp/1.0)*cp.sin(1.0*cp.pi*freq*simtime) + \
            (amp/2.0)*cp.sin(2.0*cp.pi*freq*simtime) +  \
            (amp/6.0)*cp.sin(3.0*cp.pi*freq*simtime) +  \
            (amp/3.0)*cp.sin(4.0*cp.pi*freq*simtime)).reshape(-1,)

        ft[1,:] = (cp.sin(2.0*cp.pi*freq*simtime)).reshape(-1,)
        ft[2,:] = (cp.sin(2.0*cp.pi*freq*simtime+cp.pi/2)).reshape(-1,)
        # ft[3,:] = triangle_wave(simtime).reshape(-1,)
        ft[3,:] = (cp.sin(4.0*cp.pi*freq*simtime+cp.pi/2)).reshape(-1,)
        ft2[0,:] = ((amp/1.0)*cp.sin(1.0*cp.pi*freq*simtime2) + \
            (amp/2.0)*cp.sin(2.0*cp.pi*freq*simtime2) +  \
            (amp/6.0)*cp.sin(3.0*cp.pi*freq*simtime2) +  \
            (amp/3.0)*cp.sin(4.0*cp.pi*freq*simtime2)).reshape(-1,)
        ft2[1,:] = (cp.sin(2.0*cp.pi*freq*simtime2)).reshape(-1,)
        ft2[2,:] = (cp.sin(2.0*cp.pi*freq*simtime2+cp.pi/2)).reshape(-1,)
        # ft2[3,:] = triangle_wave(simtime2).reshape(-1,)
        ft2[3,:] = (cp.sin(4.0*cp.pi*freq*simtime2+cp.pi/2)).reshape(-1,)
        #______________train_______________
        [train_out,test_out,weight_train] = nn.internal_train(None,ft,dt,alpha,nt) 
        # [train_out,test_out,weight_train] = nn.fb_train(None,ft,dt,alpha,nt)
        #(input_series,output_series,dt,alpha,nt,test_input=None)  (input_series,output_series,dt,alpha,nt,nl =0,test_input=None)
        #__transfer results format
        train_out = cp.asnumpy(train_out)
        test_out = cp.asnumpy(test_out)
        weight_train = cp.asnumpy(weight_train)
        ft = cp.asnumpy(ft)
        ft2 = cp.asnumpy(ft2)
        simtime = cp.asnumpy(simtime)
        simtime2 = cp.asnumpy(simtime2)
        end = time.time()
        print('***time consume:****',end-start)
        plt.figure()
        plt.subplot(6,1,1)
        plt.plot(simtime.T,ft.T,'b',simtime.T,train_out.T,'r')
        plt.title('training')
        plt.subplot(6,1,2)
        plt.plot(simtime.T,ft2[0,:].T,'b',simtime.T,test_out[0,:].T,'g')
        plt.title('testing1')
        plt.subplot(6,1,3)
        plt.plot(simtime.T,ft2[1,:].T,'b',simtime.T,test_out[1,:].T,'g')
        plt.title('testing2')
        plt.subplot(6,1,4)
        plt.plot(simtime.T,ft2[2,:].T,'b',simtime.T,test_out[2,:].T,'g')
        plt.title('testing3')
        plt.subplot(6,1,5)
        plt.plot(simtime.T,ft2[3,:].T,'b',simtime.T,test_out[3,:].T,'g')
        plt.title('testing4')
        plt.subplot(6,1,6)
        plt.plot(simtime.T,weight_train.T)
        plt.title('weight')
        plt.figtext(0.6, 0.01, "g={} , fb={} , Pz={} , Pgg={} \n alpha={}, N={}".format(g,fb,Pz,Pgg,alpha,N), ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.6, "pad":6})
        ct = datetime.datetime.now().strftime("%H_%M_%S")
        filename = './image/fb_output_' + ct +'.jpeg'
        plt.savefig(filename,dpi=300)
    # plt.show()

                        





    

        
