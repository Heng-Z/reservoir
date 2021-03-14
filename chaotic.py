import sys
sys.path.append('D:\\Worm\\heng\\')
from reservoir.Force import Reservoir
import matplotlib.pyplot as plt
import numpy as np
dt = 0.1
simtime = 100
nn = Reservoir(N=300,p=0.3)
run_results = nn.free_run(dt,simtime)

plt.figure();
plt.imshow(run_results)
plt.colormaps = 'jet'
plt.show()