# %%
import matplotlib.pyplot as plt
import scipy.io
# from reservoir.Force_multi_out_gpu import Reservoir,norm_percentile,norm_weight
# from reservoir.Force import Reservoir
from reservoir.Force_gpu import Reservoir
import cupy as cp
import numpy as np
import datetime
# %%
dt = 0.5
nt = 2
N = 10
alpha = 1.0 #+float(cp.random.rand(1))*0.3
fb = 2.0 #1.5 + float(cp.random.rand(1))
g = 1.5
Pz = 0.3
Pgg = 0.1
nn = Reservoir(N=N,p=Pgg,g=g)
nn.get_Jz(2,Pz,fb=fb) #(Nout,pz,fb=1)
nn.add_input(1,0.2,2)
# %%
i=0
while i<2:
    i +=1
    ft = scipy.io.loadmat('/home/heng/software/reservoir/embed_data/lorenz002.mat')['xdat']
    ft = ft/10
    ft = cp.asarray(ft.T)
    # ft = ft.T
    ft_train = ft[0:3,0:5000]
    ft_test = ft[0:3,5000:]
    inputs_train = cp.zeros((1,ft_train.shape[1]))
    inputs_test = cp.zeros((1,ft_test.shape[1]))
    
    dt = 0.5
    nt = 2
    N = 1000
    # while True:
    alpha = 1.0 #+float(cp.random.rand(1))*0.3
    fb = 2.0 #1.5 + float(cp.random.rand(1))
    g = 1.3
    Pz = 0.3
    Pgg = 0.1
    plot_range = 2000
    plot_slice = slice(2000,4000)

    nn = Reservoir(N=N,p=Pgg,g=g)
    nn.get_Jz(3,Pz,fb=fb) #(Nout,pz,fb=1)
    nn.add_input(1,0.2,2)
    # nn.change_time_coef()
    # nn.discount = 0.1
    [train_out,test_out,weight_train] = nn.fb_train(inputs_train,ft_train,dt,alpha,nt,test_input=inputs_test)
    train_out = cp.asnumpy(train_out)
    test_out = cp.asnumpy(test_out)
    weight_train = cp.asnumpy(weight_train)
    ft_train = cp.asnumpy(ft_train)
    ft_test = cp.asnumpy(ft_test)

    # plot
    plt.figure()
    plt.subplot(4,1,1)
    plt.plot(ft_train[0,:],'r',train_out[0,:],'b')
    plt.subplot(4,1,2)
    # plt.plot(ft_train[1,:],'r',train_out[1,:],'b')
    plt.subplot(4,1,3)
    plt.plot(ft_test[0,:],'r',test_out[0,:],'b')
    plt.subplot(4,1,4)
    plt.plot(weight_train.T)
    ct = datetime.datetime.now().strftime("%b_%d_%H_%M_%S")
    plt.figtext(0.6, 0.01, "g={} , fb={} , Pz={} , Pgg={} \n alpha={}, N={} \n ct:{}".format(g,fb,Pz,Pgg,alpha,N,ct), ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.6, "pad":6})
    ct = datetime.datetime.now().strftime("%b_%d_%H_%M_%S")
    filename = './image/lorenz_force_' + ct +'.jpeg'
    plt.show()
    plt.savefig(filename,dpi=300)
# %%
