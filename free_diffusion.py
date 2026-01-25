import os
import json
from datetime import datetime
import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from matplotlib import gridspec
import seaborn_image as isns

from scipy.constants import Boltzmann as kb
from hdr.oop_diffusion import *
from hdr.utils import *
from hdr.analytical_solution import *

import warnings

# ------------------------------------------------------------------------------------
# Retain each run in a safe place.
# ------------------------------------------------------------------------------------
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"free_diffusion_simulation/run_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# ------------------------------------------------------------------------------------
# CONSTANTS
# ------------------------------------------------------------------------------------


N = 100                                    # number of simultaneous simulations
dt = 0.01                                   # time step for spatial advancement
steps = 500                                 # total time steps, Time_array
D = 10 * 10**(-12)                          # Diffusion Coefficient [m^2/s]

# Can also calculate D from Stokes Einstein for Experimental Comparison
#T   = 22 + 273.15                       # K
#R_h = 50//2 * 10**(-9)                  # m
#D = kb * T / (6 * np.pi * 0.001 * R_h)  # m^2/s

print(f"Simulation Settings: \n >Particles {N} \n >Diffusion {D:.2e}")

# Save parameters
params = {
    "N": N,
    "dt": dt,
    "steps": steps,
    "D": D
}

with open(os.path.join(output_dir, "params.json"), 'w') as f:
    json.dump(params, f, indent=4)

# ------------------------------------------------------------------------------------
# Run Simulation
# ------------------------------------------------------------------------------------
sims = {}
free_sim = PeriodicCubeRandomWalk(N, dt, steps, D)
free_sim.runSimulation()
free_sim.getMeanSquaredDisplacement()
t_array_free = np.linspace(0, dt * steps, steps)
sims['free'] = (free_sim, t_array_free)

# ------------------------------------------------------------------------------------
# PLOTTING
# ------------------------------------------------------------------------------------
# ---- Layout: 3 panels on top, 1 long panel below ----
fig = plt.figure(figsize=(16, 10), dpi=250)
gs = gridspec.GridSpec(2, 2, height_ratios=[2,3], hspace=0.35, wspace=0)

ax1 = fig.add_subplot(gs[0, 0])
ax2 = fig.add_subplot(gs[1, 0])
ax3 = fig.add_subplot(gs[:, 1])
# -------------------------------------
# Trajectory PLOT
# -------------------------------------
free_sim.plotTrajectory(fig, ax1, particle_index=0)
ax1 = plt.axes(projection='3d')
ax1.set_title( "Trajectory")
# -------------------------------------
# Displacement Histogram PLOT
# -------------------------------------
XD = free_sim.TRAJ
x,y,z = XD[0]
ax2.hist(x)
ax2.hist(y)
ax2.hist(z)
# -------------------------------------
# FREE MSD PLOT
# -------------------------------------
free_sim, t_cyl = sims['free']
free_sim.plotMeanMSD(fig, ax3)
ax3.set_title( "MSD")
ax3.set_xlabel("Time")
ax3.set_ylabel("$<x^2>$")
ax3.grid(True, which='both', linestyle=':', linewidth=0.5)
ax3.legend(loc='best', frameon=False)

plt.show()