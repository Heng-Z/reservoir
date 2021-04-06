import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import datetime
import cupy as cp
from reservoir.utils import conv_gauss
from reservoir.Force_multi_out_gpu import Reservoir
# %%
ft = scipy.io.loadmat('./embed_data/crawl_icasig_driven.mat')['crawl_icasig_driven']
inputs = ft[:,7]
inputs = np.roll(inputs,-5) #+ np.random.randn(*inputs.shape) *0.1
inputs0 = np.ones(inputs.shape) - inputs
inputs_conv = conv_gauss(inputs).reshape(-1,1)
inputs_conv0 = conv_gauss(inputs0).reshape(-1,1)
drive_sig = np.hstack([inputs_conv,inputs_conv0])
y_train = ft[0:20000,[2,3,5,6]] 
y_train = y_train/np.array([1,1,2,2])
y_test = ft[20000:30000,[2,3,5,6]]
y_test = y_test/np.array([1,1,2,2])
x_train = drive_sig[0:20000,:]
x_test = drive_sig[20000:30000,:]
Nout = 4

y_train = cp.asarray(y_train.T)
x_train = cp.asarray(x_train.T)
y_test = cp.asarray(y_test.T)
x_test = cp.asarray(x_test.T)
# %%
while True:
    nt = 2
    N = 1000
    # while True:
    # %%
    alpha = 1+float(cp.random.rand(1))*0.3
    fb = 1.5 + float(cp.random.rand(1))
    g = 1.6
    Pz = 0.1
    Pgg = 0.1
    plot_range = 2000
    plot_slice = slice(2000,4000)
    dt = 0.1

    nn = Reservoir(N=N,p=Pgg,g=g)
    nn.get_Jz(Nout,Pz,fb=fb) #(Nout,pz,fb=1)
    nn.add_input(2,0.2,2)

    [train_out,test_out,weight_train] = nn.fb_train(x_train,y_train,dt,alpha,nt,nl=0.1,test_input=x_test)

    train_out = cp.asnumpy(train_out)
    test_out = cp.asnumpy(test_out)
    weight_train = cp.asnumpy(weight_train)
    y_train = cp.asnumpy(y_train)
    y_test = cp.asnumpy(y_test)
    y_pred = test_out

    plot_slice = slice(1000,6000)
    plt.figure()
    plt.subplot(4,1,1)
    plt.plot(y_train[0,plot_slice].T,'r',train_out[0,plot_slice].T,'b')
    plt.subplot(4,1,2)
    plt.plot(y_train[2,plot_slice].T,'r',train_out[2,plot_slice].T,'b')
    plt.subplot(4,1,3)
    plt.plot(y_test[0,plot_slice].T,'r',y_pred[0,plot_slice].T,'b')
    plt.subplot(4,1,4)
    plt.plot(y_test[2,plot_slice].T,'r',y_pred[2,plot_slice].T,'b')
    ct = datetime.datetime.now().strftime("%b_%d_%H_%M_%S")
    # plt.figtext(0.6, 0.01, "leaky: {}, radius: {} \n fb: {} ridge: {} \n {}".format(leakingRate,spectralRadius,feedbackScaling,regressionParameters,ct), ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.6, "pad":6})
    plt.figtext(0.6, 0.01, "g={} , fb={} , Pz={} , Pgg={} \n alpha={}, N={} \n {}".format(g,fb,Pz,Pgg,alpha,N,ct), ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.6, "pad":6})
    filename = './image/worm_force_driven' + ct +'.jpeg'
    # plt.show()
    plt.savefig(filename,dpi=300)
    # esn.save('./esns/worm_esn_driven' + ct+'.net')