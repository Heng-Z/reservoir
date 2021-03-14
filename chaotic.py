import sys
sys.path.append('D:\\Worm\\heng\\')
from reservoir.Force import Reservoir
import matplotlib.pyplot as plt
import numpy as np
dt = 0.1
time_sec = 400
nn = Reservoir(N=300,p=0.3,g=2)
run_results = nn.free_run(dt,time_sec)

dt = 0.1
simtime = np.arange(0,time_sec,step=dt).reshape(1,-1)
simtime2 = np.arange(time_sec,2*time_sec,step=dt).reshape(1,-1)
amp = 1.3
freq = 1/60
ft = (amp/1.0)*np.sin(1.0*np.pi*freq*simtime) + \
    (amp/2.0)*np.sin(2.0*np.pi*freq*simtime) +  \
    (amp/6.0)*np.sin(3.0*np.pi*freq*simtime) +  \
    (amp/3.0)*np.sin(4.0*np.pi*freq*simtime)
ft = ft.reshape(1,-1)

plt.figure()
plt.subplot(5,1,1)
plt.plot(run_results[0,:])
plt.subplot(5,1,2)
plt.plot(run_results[1,:])
plt.subplot(5,1,3)
plt.plot(run_results[2,:])
plt.subplot(5,1,4)
plt.plot(run_results[3,:])
plt.subplot(5,1,5)
plt.plot(ft.T)

plt.show()