import scipy.io
import numpy as np
import matplotlib.pyplot as plt
import cupy as cp
import time
import datetime
from reservoir.Force_multi_out_gpu import Reservoir,norm_percentile,norm_weight
matfile = './embed_data/roam_real_ica.mat'
signals = scipy.io.loadmat(matfile)
start = time.time()
ct = datetime.datetime.now().strftime("%H_%M_%S")
time_sec = 1440
dt = 0.1
nt = 2
N = 10
alpha = 1
fb =2.0
g = 1.6
Pz = 0.4
Pgg = 0.5
plot_range = 2000
nn = Reservoir(N=N,p=Pgg,g=g)
nn.get_Jz(2,Pz,fb=fb) #(Nout,pz,fb=1)
# simtime = cp.arange(0,time_sec,step=dt).reshape(1,-1)
# simtime2 = cp.arange(time_sec,2*time_sec,step=dt).reshape(1,-1)
inputs = cp.asarray(signals['roam_real_ica'])[2:4:,20:-1]
# inputs = inputs[:,1:-1:3]
# inputs = norm_percentile(inputs,pcnt=False)
# inputs = inputs/2
inputs = norm_weight(inputs,[1,1])
print('*****',inputs.shape)
simtime = cp.arange(plot_range).reshape(1,-1)
[train_out,test_out,weight_train] = nn.fb_train(None,inputs,dt,alpha,nt,nl=0)
train_out = cp.asnumpy(train_out)
test_out = cp.asnumpy(test_out)
weight_train = cp.asnumpy(weight_train)
simtime = cp.asnumpy(simtime)
inputs = cp.asnumpy(inputs)
end = time.time()
print('***time consume:****',end-start)
plt.figure()
# print(test_out[0,0:500])
plt.subplot(5,1,1)
plt.plot(simtime.T,inputs[0:2,0:plot_range].T,'b',simtime.T,train_out[0:2,0:plot_range].T,'r')
plt.title('training')
plt.subplot(5,1,2)
# plt.plot(simtime.T,inputs[2:4,0:plot_range].T,'b',simtime.T,train_out[2:4,0:plot_range].T,'r')
# plt.title('training 5-7')
# plt.subplot(5,1,3)
plt.plot(simtime.T,test_out[0:2,0:plot_range].T,'g')
plt.title('testing1')
plt.subplot(5,1,4)
# plt.plot(simtime.T,test_out[2:4,0:plot_range].T,'g')
# plt.title('testing2')
# plt.subplot(5,1,5)
plt.plot(simtime.T,weight_train[:,0:plot_range].T)
plt.title('weight')
plt.figtext(0.6, 0.01, "g={} , fb={} , Pz={} , Pgg={} \n alpha={}, N={}".format(g,fb,Pz,Pgg,alpha,N), ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.6, "pad":6})
ct = datetime.datetime.now().strftime("%H_%M_%S")
filename = './image/worm_' + ct +'.jpeg'
plt.savefig(filename,dpi=300)
np.save('net_params_'+ ct +'.npz',nn.Jz,nn.Jgz,nn.M)
