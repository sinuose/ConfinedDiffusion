import os
import json
from datetime import datetime
import numpy as np


import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

from scipy.constants import Boltzmann as kb
from hdr.oop_diffusion import *
from hdr.utils import *
from hdr.analytical_solution import *



if __name__ == "__main__":
    # ------------------------------------------------------------------------------------
    # Retain each run in a safe place.
    # ------------------------------------------------------------------------------------
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"simulation_output/run_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # ------------------------------------------------------------------------------------
    # CONSTANTS
    # ------------------------------------------------------------------------------------


    N = 1000                                      # number of simultaneous simulations
    dt = 0.01                                    # time step for spatial advancement
    steps = 1000                                 # total time steps, Time_array
    D = 5 * 10**(-12)                            # Diffusion Coefficient [m^2/s]
    R = 3 * 10**(-6)                             # Cylinder Radius       [m]
    H = 3 * 10**(-6)                             # Cylinder Height       [m]

    # Can also calculate D from Stokes Einstein for Experimental Comparison
    #T   = 22 + 273.15                       # K
    #R_h = 50//2 * 10**(-9)                  # m
    #D = kb * T / (6 * np.pi * 0.001 * R_h)  # m^2/s

    print(f"Simulation Settings: \n >Particles {N} \n >Radius {R:.2e} \n >Height {H:.2e} \n >Diffusion {D:.2e}")

    # Save parameters
    params = {
        "N": N,
        "dt": dt,
        "steps": steps,
        "D": D,
        "R": R,
        "H": H
    }
    with open(os.path.join(output_dir, "params.json"), 'w') as f:
        json.dump(params, f, indent=4)

    # ------------------------------------------------------------------------------------
    # RUN ALL 3 SIMULATIONS
    # ------------------------------------------------------------------------------------
    sims = {}

    # Cylinder
    sim_cyl = CylinderRandomWalk(N, dt, steps, D, R, H)
    sim_cyl.runSimulation()
    sim_cyl.getMeanSquaredDisplacement()
    t_array_cyl = np.linspace(0, dt * steps, steps)
    ana_cyl = CylinderAnalyticalMSDGPT(D, R, H, t_array_cyl)
    sims['cylinder'] = (sim_cyl, t_array_cyl, ana_cyl)

    # Circle
    sim_circ = CircleRandomWalk(N, dt, steps, D, R)
    sim_circ.runSimulation()
    sim_circ.getMeanSquaredDisplacement()
    t_array_circ = np.linspace(0, dt * steps, steps)
    ana_circ = CircleAnalyticalMSD(D, R, t_array_circ)
    sims['circle'] = (sim_circ, t_array_circ, ana_circ)

    # Line
    sim_line = LineRandomWalk(N, dt, steps, D, H)
    sim_line.runSimulation()
    sim_line.getMeanSquaredDisplacement()
    t_array_line = np.linspace(0, dt * steps, steps)
    ana_line = LineAnalyticalMSD(D, H, t_array_line)
    sims['line'] = (sim_line, t_array_line, ana_line)

    # ------------------------------------------------------------------------------------
    # ERROR ANALYSIS
    # ------------------------------------------------------------------------------------
    # ideally this should be comparing the experimental to the analytical

    # ------------------------------------------------------------------------------------
    # PLOTTING
    # ------------------------------------------------------------------------------------
    fig, axs = plt.subplots(3, 1, figsize=(6, 10), constrained_layout=True)

    labelsize = 10
    titlesize = 12
    legendsize = 8

    # Helper to save MSD data
    def save_msd_data(name, t, msd):
        np.savetxt(os.path.join(output_dir, f"{name}_msd.csv"),
                np.column_stack((t, msd)), delimiter=",", header="Time,MSD", comments='')

    # -------------------------------------
    # CYLINDER MSD PLOT
    # -------------------------------------
    ax_cyl = axs[2]
    sim_cyl, t_cyl, ana_cyl = sims['cylinder']
    sim_cyl.plotMeanMSD(fig, ax_cyl)
    ax_cyl.plot(t_cyl, ana_cyl, linestyle='--', color='black', label="Analytical")
    save_msd_data("cylinder", t_cyl, ana_cyl)

    ax_cyl.set_title("Cylinder MSD", fontsize=titlesize)
    ax_cyl.set_xlabel("Time", fontsize=labelsize)
    ax_cyl.set_ylabel("MSD", fontsize=labelsize)
    ax_cyl.grid(True, which='both', linestyle=':', linewidth=0.5)
    ax_cyl.legend(fontsize=legendsize, loc='lower right', frameon=False)

    # Inset: Cylinder Trajectory
    inset_position = [0.75, 0.122 , 0.2, 0.2]  # [left, bottom, width, height] in figure coordinates
    inset_cyl = fig.add_axes(inset_position, projection='3d')
    inset_cyl.set_box_aspect([1, 1, 1])  # Equal aspect ratio
    inset_cyl.view_init(elev=30, azim=45)  # Isometric view

    Xc, Yc, Zc = data_for_cylinder_along_z(0, 0, R, H)
    inset_cyl.plot_surface(Xc, Yc, Zc, alpha=0.15, color='gray', edgecolor='none')

    for i in range(1):
       sim_cyl.plotTrajectory(fig, inset_cyl, particle_index=i)

    inset_cyl.set_xticks([])
    inset_cyl.set_yticks([])
    inset_cyl.set_zticks([])
    inset_cyl.set_box_aspect([1, 1, 0.5])  # vertical cylinder
    inset_cyl.view_init(elev=20, azim=45)
    inset_cyl.grid(False)
    inset_cyl.set_facecolor((0, 0, 0, 0))  # transparent background

    # -------------------------------------
    # CIRCLE MSD PLOT
    # -------------------------------------
    ax_circ = axs[1]
    sim_circ, t_circ, ana_circ = sims['circle']
    sim_circ.plotMeanMSD(fig, ax_circ)
    ax_circ.plot(t_circ, ana_circ, linestyle='--', color='black', label="Analytical")
    save_msd_data("circle", t_circ, ana_circ)

    ax_circ.set_title("Circle MSD", fontsize=titlesize)
    ax_circ.set_xlabel("Time", fontsize=labelsize)
    ax_circ.set_ylabel("MSD", fontsize=labelsize)
    ax_circ.grid(True, linestyle=':', linewidth=0.5)
    ax_circ.legend(fontsize=legendsize, loc='lower right', frameon=False)

    # Inset: Circle Trajectory (2D)
    inset_circ = inset_axes(ax_circ, width="30%", height="30%", loc='upper right', borderpad=1)
    inset_circ.set_aspect('equal')
    for i in range(1):
        sim_circ.plotTrajectory(fig, inset_circ, particle_index=i)
    inset_circ.set_xticks([])
    inset_circ.set_yticks([])
    inset_circ.set_facecolor((0, 0, 0, 0))

    # -------------------------------------
    # LINE MSD PLOT
    # -------------------------------------
    ax_line = axs[0]
    sim_line, t_line, ana_line = sims['line']
    sim_line.plotMeanMSD(fig, ax_line)
    ax_line.plot(t_line, ana_line, linestyle='--', color='black', label="Analytical")
    save_msd_data("line", t_line, ana_line)

    ax_line.set_title("Line MSD", fontsize=titlesize)
    ax_line.set_xlabel("Time", fontsize=labelsize)
    ax_line.set_ylabel("MSD", fontsize=labelsize)
    ax_line.grid(True, linestyle=':', linewidth=0.5)
    ax_line.legend(fontsize=legendsize, loc='lower right', frameon=False)

    # Inset: Line Trajectory (1D)
    inset_line = inset_axes(ax_line, width="30%", height="30%", loc='upper right', borderpad=1)
    for i in range(1):
        sim_line.plotTrajectory(fig, inset_line, particle_index=i)
    inset_line.set_xticks([])
    inset_line.set_yticks([])
    inset_line.set_facecolor((0, 0, 0, 0))

    # -------------------------------------
    # Save and Display
    # -------------------------------------
    # Save SVG for LaTeX
    svg_path = os.path.join(output_dir, f"{timestamp}.svg")
    fig.savefig(svg_path, format='svg')

    plt.suptitle("Mean Squared Displacement for Confined Random Walks", fontsize=14)
    plt.show()