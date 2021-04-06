import sys
from reservoir.Force_multi_out_gpu import Reservoir
import matplotlib.pyplot as plt
import numpy as np
import cupy as cp
wash = 0.5
time_sec = 40000
nn = Reservoir(N=1000,p=0.1,g=1.4)
nn.get_Jz(4,0.2,fb=2.0)
run_results = nn.free_esn(wash,time_sec)
print('spectrual radius:',cp.max(cp.abs(cp.linalg.eigh(nn.M)[0])))

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