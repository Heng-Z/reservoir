import sys
sys.path.append('D:\\Worm\\heng\\')
from reservoir.Force_multi_out_gpu import Reservoir
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
dt = 0.1
time_sec = 4000
nn = Reservoir(N=1000,p=0.1,g=3)
nn.get_Jz(4,0.2,fb=2.0)
run_results = nn.free_run(dt,time_sec)
print('spectrual radius:',cp.max(cp.abs(cp.linalg.eigh(nn.M)[0])))
# dt = 0.1
# simtime = np.arange(0,time_sec,step=dt).reshape(1,-1)
# simtime2 = np.arange(time_sec,2*time_sec,step=dt).reshape(1,-1)
# amp = 1.3
# freq = 1/60
# ft = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime) + \
#     (amp/2.0)*np.sin(2.0*np.pi*freq*simtime) +  \
#     (amp/6.0)*np.sin(3.0*np.pi*freq*simtime) +  \
#     (amp/3.0)*np.sin(4.0*np.pi*freq*simtime)
# ft = ft.reshape(1,-1)
to_plot = cp.asnumpy(run_results[0:5,:]) #
to_plot_f = np.abs(np.fft.fftshift(np.fft.fft(to_plot,axis=1)))
plt.figure()
plt.subplot(5,1,1)
plt.plot(to_plot_f[0,:])
plt.subplot(5,1,2)
plt.plot(to_plot_f[1,:])
plt.subplot(5,1,3)
plt.plot(to_plot_f[2,:])
plt.subplot(5,1,4)
plt.plot(to_plot_f[3,:])
plt.subplot(5,1,5)
plt.plot(to_plot_f[4,:])
plt.figure()
plt.plot(to_plot[0,:])
plt.subplot(5,1,2)
plt.plot(to_plot[1,:])
plt.subplot(5,1,3)
plt.plot(to_plot[2,:])
plt.subplot(5,1,4)
plt.plot(to_plot[3,:])
plt.subplot(5,1,5)
plt.plot(to_plot[4,:])

plt.show()