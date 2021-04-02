# %%
from easyesn import PredictionESN
from easyesn.optimizers import GradientOptimizer,GridSearchOptimizer
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
# %%
ft = scipy.io.loadmat('./embed_data/crawl_tr5.mat')['crawl5']
ft = (ft - np.mean(ft,axis = 0))/np.std(ft,axis=0)
plt.plot(ft)
inputs = np.random.randn(ft.shape[0],1)*0.1
y_train = ft[0:20000,:]
y_test = ft[20000:30000,:]
x_train = inputs[0:20000,:]
x_test = inputs[20000:30000,:]
# %%
esn = PredictionESN(n_input=1,n_output=5,n_reservoir=1000,leakingRate=0.7,
spectralRadius=1.5,regressionParameters=[1e-2],solver='lsqr',
feedback=True,feedbackScaling=2)
esn.fit(x_train,y_train,transientTime=500,verbose=1)
y_pred = esn.predict(x_test)
plot_slice = slice(1000,2000)
plt.subplot(4,1,1)
plt.plot(y_pred[plot_slice,0],'r',y_test[plot_slice,0],'b')
plt.subplot(4,1,2)
plt.plot(y_pred[plot_slice,1],'r',y_test[plot_slice,1],'b')
plt.subplot(4,1,3)
plt.plot(y_pred[plot_slice,2],'r',y_test[plot_slice,2],'b')
plt.subplot(4,1,4)
plt.plot(y_pred[plot_slice,3],'r',y_test[plot_slice,3],'b')
# %%
