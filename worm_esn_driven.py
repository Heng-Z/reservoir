# %%
from easyesn import PredictionESN
from easyesn.optimizers import GradientOptimizer,GridSearchOptimizer
import numpy as np
import matplotlib.pyplot as plt
import scipy.io
import datetime
from reservoir.Force import sprandn
from reservoir.utils import conv_gauss
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
# y_train = np.hstack([y_train,x_train])
# y_test = np.hstack([y_test,x_test])
# x_train = np.random.randn(*x_train.shape)*0.1
# x_test = np.random.randn(*x_test.shape) *0.1
sparse_feedback = sprandn(1000,Nout+1,0.2) # +1 is for outputbias
# %%
while True:
    leakingRate = np.random.rand(1)
    # leakingRate = 0.07 + 0.02*np.random.randn(1)
    spectralRadius = 0.5 + np.random.rand(1)
    # spectralRadius = 1.04+ 0.07*np.random.randn(1)
    feedbackScaling = 1.5 + np.random.randn(1)
    # feedbackScaling = 2
    regressionParameters = 0.1
    esn = PredictionESN(n_input=2,n_output=Nout,n_reservoir=1000,leakingRate=leakingRate,
    spectralRadius=spectralRadius,regressionParameters=[1e-1],solver='lsqr',
    feedback=True,feedbackScaling=feedbackScaling,inputDensity=0.1)
    esn.resetFeedbackMatrix(sparse_feedback*feedbackScaling)
    esn.fit(x_train,y_train,transientTime=1000,verbose=1)
    y_pred = esn.predict(x_test)
    # opt = GradientOptimizer(esn, learningRate=0.001)

    # validationLosses, fitLosses, inputScalings, spectralRadiuses, leakingRates, learningRates = opt.optimizeParameterForValidationError(x_train, y_train, x_test, y_test, epochs=20, transientTime=1000)
    # print(validationLosses, fitLosses, inputScalings, spectralRadiuses, leakingRates, learningRates)
    # %%
    plot_slice = slice(1000,6000)
    plt.figure()
    plt.subplot(2,1,1)
    plt.plot(y_pred[plot_slice,0],'r',y_test[plot_slice,0],'b')
    plt.subplot(2,1,2)
    plt.plot(y_pred[plot_slice,2],'r',y_test[plot_slice,2],'b')
    ct = datetime.datetime.now().strftime("%b_%d_%H_%M_%S")
    plt.figtext(0.6, 0.01, "leaky: {}, radius: {} \n fb: {} ridge: {} \n {}".format(leakingRate,spectralRadius,feedbackScaling,regressionParameters,ct), ha="center", fontsize=12, bbox={"facecolor":"orange", "alpha":0.6, "pad":6})
    filename = './image/worm_esn_driven' + ct +'.jpeg'
    # plt.show()
    plt.savefig(filename,dpi=300)
    esn.save('./esns/worm_esn_driven' + ct+'.net')
# %%

