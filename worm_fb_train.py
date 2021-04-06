# %%
import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import time
import datetime
import sys
sys.path.append('/home/heng/software/')
from reservoir.Force_multi_out_gpu import Reservoir,norm_percentile,norm_weight
matfile = './embed_data/crawl_icasig_driven.mat'
signals = scipy.io.loadmat(matfile)
while True:
    start = time.time()
    ct = datetime.datetime.now().strftime("%H_%M_%S")
    time_sec = 1440
    dt = 0.1
    nt = 2
    N = 1000
    # while True:
    # %%
    alpha = 1+float(cp.random.rand(1))*0.3
    fb = 1.5 + float(cp.random.rand(1))
    g = 4
    Pz = 0.1
    Pgg = 0.1
    plot_range = 2000
    plot_slice = slice(2000,4000)
    # simtime = cp.arange(0,time_sec,step=dt).reshape(1,-1)
    # simtime2 = cp.arange(time_sec,2*time_sec,step=dt).reshape(1,-1)
    inputs = cp.asarray(signals['crawl_icasig_driven'])[:,0:7].T
    drive = cp.asarray(signals['crawl_icasig_driven'])[:,7:].T
    L = inputs.shape[1]
    inputs = inputs[:,0:L/5]
    drive1 = drive[:,0:L/5]
    drive2 = (cp.ones(drive1.shape)-drive1)*2

    drive_sig = cp.vstack([drive1,drive2])
    inputs = norm_weight(inputs,[5,5,2,2,5,5,5])
    print('*****',inputs.shape)
    simtime = cp.arange(plot_range).reshape(1,-1)

    nn = Reservoir(N=N,p=Pgg,g=g)
    nn.get_Jz(7,Pz,fb=fb) #(Nout,pz,fb=1)
    nn.add_input(2,0.2,2)
    # nn.change_time_coef()
    # nn.discount = 0.1
    [train_out,test_out,weight_train] = nn.fb_train(drive_sig,inputs,dt,alpha,nt,nl=0.1)
    train_out = cp.asnumpy(train_out)
    test_out = cp.asnumpy(test_out)
    weight_train = cp.asnumpy(weight_train)
    simtime = cp.asnumpy(simtime)
    inputs = cp.asnumpy(inputs)
    end = time.time()
    print('***time consume:****',end-start)
    # %%
    plt.figure()
    # print(test_out[0,0:500])
    plt.subplot(5,1,1)
    plt.plot(simtime.T,inputs[2:4,plot_slice].T,'b',simtime.T,train_out[2:4,plot_slice].T,'r')
    plt.title('training')
    plt.subplot(5,1,2)
    plt.plot(simtime.T,inputs[5:7,plot_slice].T,'b',simtime.T,train_out[5:7,plot_slice].T,'r')
    plt.title('training 6-7')
    plt.subplot(5,1,3)
    plt.plot(simtime.T,test_out[2:4,plot_slice].T,'g')
    plt.title('testing1')
    plt.subplot(5,1,4)
    plt.plot(simtime.T,test_out[5:7,plot_slice].T,'g')
    plt.title('testing2')
    plt.subplot(5,1,5)
    plt.plot(range(3*plot_range),weight_train[:,0:3*plot_range].T)
    plt.title('weight')
    plt.figtext(0.6, 0.01, "g={} , fb={} , Pz={} , Pgg={} \n alpha={}, N={}".format(g,fb,Pz,Pgg,alpha,N), ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.6, "pad":6})
    ct = datetime.datetime.now().strftime("%b_%d_%H_%M_%S")
    filename = './image/fb_worm_driven' + ct +'.jpeg'
    plt.show()
    plt.savefig(filename,dpi=300)
    # np.save('net_params_'+ ct +'.npz',nn.Jz,nn.Jgz,nn.M)

    # %%
