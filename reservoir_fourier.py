import sys
sys.path.append('D:\\Worm\\heng\\')
from reservoir.Force import Reservoir
import matplotlib.pyplot as plt
import numpy as np
dt = 0.1
time_sec = 4000
nn = Reservoir(N=1000,p=0.1,g=2)
run_results = nn.free_run(dt,time_sec)
