# %%
from easyesn import PredictionESN
from easyesn.optimizers import GradientOptimizer,GridSearchOptimizer
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
# %%
from easyesn import optimizers
import importlib
importlib.reload(optimizers)

# %%
# fl = np.load('./embed_data/longtime_train.npz')
# y_train = fl['arr_0']
# x_train = fl['arr_1']
# y_test = y_train[:,500:4500]
# x_test = x_train[:,500:4500]
ft = scipy.io.loadmat('/home/heng/software/easyesn/mytest/lorenz_data.mat')['xdat']
ft = (ft - np.mean(ft,axis = 0))/np.std(ft,axis=0)
plt.plot(ft)
inputs = np.zeros((ft.shape[0],1))
y_train = ft[0:4500,:]
y_test = ft[4500:6500,:]
x_train = inputs[0:4500,:]
x_test = inputs[4500:6500,:]
# %%
esn = PredictionESN(n_input=1,n_output=3,n_reservoir=500,leakingRate=0.2,spectralRadius=1.2,regressionParameters=[1e-2],solver='lsqr',feedback=True)
esn.fit(x_train,y_train,transientTime=100,verbose=1)
y_pred = esn.predict(x_test)
plt.plot(y_pred[:,1],'r',y_test[:,1],'b')
# %%
opt = optimizers.GradientOptimizer(esn,learningRate=0.001)
validationLosses, fitLosses, inputScalings, spectralRadiuses, leakingRates, learningRates = opt.optimizeParameterForTrainError(x_train, y_train[:,0:1], x_test, y_test[:,0:1], epochs=150, transientTime=100)
# %%
matfile = './embed_data/roam_real_ica.mat'
signals = scipy.io.loadmat(matfile)
ft = signals['roam_real_ica'].T
ft = ft/np.array([5,5,2,2,5,5,5])
inputs = np.random.randn(ft.shape[0],1)
y_train = ft[0:20000,2:4]
y_test = ft[20000:25000,2:4]
x_train = inputs[0:20000,:]
x_test = inputs[20000:25000,:]
# %%
esn = PredictionESN(n_input=1,n_output=2,n_reservoir=1000,leakingRate=0.2,spectralRadius=1.0,regressionParameters=[1e-2],solver='lsqr',feedback=True)
esn.fit(x_train,y_train,transientTime=500,verbose=1)
y_pred = esn.predict(x_test)
plt.subplot(4,1,1)
plt.plot(y_pred[:,2],'r',y_test[:,2],'b')
plt.subplot(4,1,2)
plt.plot(y_pred[:,3],'r',y_test[:,3],'b')
plt.subplot(4,1,3)
plt.plot(y_pred[:,5],'r',y_test[:,5],'b')
plt.subplot(4,1,4)
plt.plot(y_pred[:,6],'r',y_test[:,6],'b')
# %%
