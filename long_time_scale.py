# %%
import numpy as np
from numpy import matlib
import matplotlib.pyplot as plt
import cupy as cp
from reservoir import Force_multi_out_gpu 
import importlib
import datetime
time_sec = 200
dt = 0.1
simtime = cp.arange(0,time_sec,step=dt).reshape(1,-1)
simtime2 = cp.arange(time_sec,2*time_sec,step=dt).reshape(1,-1)
amp = 1.3
freq = 1/4
ft = cp.zeros((2,simtime.shape[1]))
ft2 = cp.zeros((2,simtime2.shape[1]))
ft[0,:] = ((amp/1.0)*cp.sin(1.0*cp.pi*freq*simtime) + \
    (amp/2.0)*cp.sin(2.0*cp.pi*freq*simtime) +  \
    (amp/6.0)*cp.sin(3.0*cp.pi*freq*simtime) +  \
    (amp/3.0)*cp.sin(4.0*cp.pi*freq*simtime)).reshape(-1,)


intv = int(2/freq/dt)

flt = cp.ones(simtime.shape)
flt[0,2*intv:3*intv]=0.1
flt[0,6*intv:7*intv]=0.1
flt[0,9*intv:10*intv]=0.1
flt[0,14*intv:15*intv]=0.1
flt[0,21*intv:22*intv]=0.1
ft0 = cp.multiply(flt.reshape(-1,),ft[0])/1.8
tri_wave = cp.zeros(intv)
tri_wave[0:intv//2] = cp.arange(intv//2)/(intv//2)
tri_wave[intv//2:intv-1] = cp.arange(intv//2)[-1:0:-1]/(intv//2)

ft1 = cp.zeros((1,simtime.shape[1]))

ft1[0,2*intv:3*intv]=tri_wave.reshape(-1,)
ft1[0,6*intv:7*intv]=tri_wave.reshape(-1,)
ft1[0,9*intv:10*intv]=tri_wave.reshape(-1,)
ft1[0,14*intv:15*intv]=tri_wave.reshape(-1,)
ft1[0,21*intv:22*intv]=tri_wave.reshape(-1,)

ft[0] = ft0
ft[1] = ft1[0]
ft = cp.hstack([ft,ft,ft,ft,ft])
inputs0 = (1/flt -1 )/9
inputs1 = (1-inputs0)*2
# inputs = cp.vstack([inputs0,inputs1])
inputs = cp.ones((1,2000))*0.1
inputs = cp.hstack([inputs,inputs,inputs,inputs,inputs])
test_inputs = cp.flip(inputs,axis=1)

np.savez('./embed_data/longtime_train.npz',cp.asnumpy(ft),cp.asnumpy(inputs))
# np.savez('longtime_test.npz',ft,inpu)
# %%
while True:
    nt = 3
    N = 500
    alpha = 1+ float(np.random.randn(1))
    fb =1.0 + float(np.random.rand(1))
    g = 1.6
    gin = 1.0 + float(np.random.rand(1))
    Pz = 0.2
    # Pgg = float(cp.random.rand(1))*0.75
    Pgg = 0.1
    nn = Force_multi_out_gpu.Reservoir(N=N,p=Pgg,g=g)
    nn.get_Jz(2,Pz,fb=fb) #(Nout,pz,fb=1)
    nn.add_input(2,0.3,gin)
    # nn.change_time_coef()
    [train_out,test_out,weight_train] = nn.fb_train(inputs,ft,dt,alpha,nt,nl =0.1) 

    train_out = cp.asnumpy(train_out)
    test_out = cp.asnumpy(test_out)
    weight_train = cp.asnumpy(weight_train)
    ft = cp.asnumpy(ft)
    ft2 = cp.asnumpy(ft2)
    simtime = cp.asnumpy(simtime)
    simtime2 = cp.asnumpy(simtime2)

    prange = slice(8000,10000)
    plt.figure()
    plt.plot(weight_train.T)
    plt.figure()
    plt.subplot(5,1,1)
    plt.plot(ft[0,prange].T,'b',train_out[0,prange].T,'r')
    plt.subplot(5,1,2)
    plt.plot(ft[1,prange].T,'b',train_out[1,prange].T,'r')
    plt.subplot(5,1,3)
    plt.plot(ft[0,prange].T,'b',test_out[0,prange].T,'r')
    plt.subplot(5,1,4)
    plt.plot(ft[1,prange].T,'b',test_out[1,prange].T,'r')
    plt.subplot(5,1,5)
    plt.plot(weight_train[:,0:-1].T)
    plt.figtext(0.6, 0.01, "g={} , fb={} , Pz={} , Pgg={} \n alpha={}, N={}".format(g,fb,Pz,Pgg,alpha,N), ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.6, "pad":6})
    ct = datetime.datetime.now().strftime("%b_%d_%H_%M_%S")
    filename = './image/long_time_pulse_inp_' + ct +'.jpeg'
    plt.savefig(filename,dpi=300)
# %%
